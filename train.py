import argparse
import datetime
import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import weaklyCD
from utils import evaluate_CD, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils_CD import cam_to_label,multi_scale_cam
from utils.optimizer import PolyWarmupAdamW
from models.model import TransWCD_dual, TransWCD_single
from DISep.high_pass_inst_retrie import hp_inst_retrie
from DISep.disep_loss import DISEP_Loss

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# LEVIR/DSIFN/WHU.yaml
parser.add_argument("--config",
                    default='configs/SYSU.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--crop_size", default=256, type=int, help="crop_size")
parser.add_argument("--scheme", default='transwcd_single', type=str, help="transwcd_dual or transwcd_single")
parser.add_argument('--pretrained', default= True, type=bool, help="pretrained")
parser.add_argument('--checkpoint_path', default= False, type=str, help="checkpoint_path" )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w

def validate(model=None, data_loader=None, cfg=None):
    enc_preds, gts, cams = [], [], []
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs_A, inputs_B, labels, cls_label = data

            inputs_A = inputs_A.cuda()
            inputs_B = inputs_B.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()


            _cams = multi_scale_cam(model, inputs_A, inputs_B, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            _, cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)


            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))


    cam_score = evaluate_CD.scores(gts, cams)


    model.train()
    return cam_score, labels




def train(cfg):
    num_workers = 10

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = weaklyCD.ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        num_classes=cfg.dataset.num_classes,
    )

    val_dataset = weaklyCD.CDDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.batch_size,
                              # shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)
    device = torch.device('cuda')

    if cfg.scheme == "transwcd_dual":
        transwcd = TransWCD_dual(backbone=cfg.backbone.config,
                                 stride=cfg.backbone.stride,
                                 num_classes=cfg.dataset.num_classes,
                                 embedding_dim=256,
                                 pretrained=args.pretrained,
                                 pooling=args.pooling, )
    elif cfg.scheme == "transwcd_single":
        transwcd = TransWCD_single(backbone=cfg.backbone.config,
                                 stride=cfg.backbone.stride,
                                 num_classes=cfg.dataset.num_classes,
                                 embedding_dim=256,
                                 pretrained=args.pretrained,
                                 pooling=args.pooling, )
    else:
        print("Please choose a baseline structure in /configs/...yaml")

    #logging.info('\nNetwork config: \n%s' % (transwcd))
    param_groups = transwcd.get_param_groups()
    transwcd.to(device)

    writer = SummaryWriter(cfg.work_dir.logger_dir)
    print('writer:',writer)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,  ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()



    best_F1 = 0.0
    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs_A, inputs_B, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs_A, inputs_B, cls_labels, img_box = next(train_loader_iter)

        inputs_A = inputs_A.to(device, non_blocking=True)
        inputs_B = inputs_B.to(device, non_blocking=True)

        cls_labels = cls_labels.to(device, non_blocking=True)


        cams = multi_scale_cam(transwcd, inputs_A=inputs_A, inputs_B=inputs_B, scales=cfg.cam.scales)


        valid_cam, cam_label = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True, cfg=cfg)

        cls, enc_cd = transwcd(inputs_A, inputs_B)

        enc_cd = F.interpolate(enc_cd, size=valid_cam.shape[2:], mode='bilinear', align_corners=False)

### DISep Module ###
        hp_cam, centriods, feat_inst, mask_inst = hp_inst_retrie(cams.detach(), feat= enc_cd, cls_label=cls_labels, img_box=img_box, ignore_mid=True,
                                                      cfg=cfg)

        loss_disep = DISEP_Loss()                                       # margin_uc_c=0.0, margin_c_c=0.0
        disep_loss, _ = loss_disep(centriods, feat_inst, mask_inst, cls_labels)  #, img_loss, triplet_loss, contrastive_loss


        # change classification loss
        cc_loss = F.multilabel_soft_margin_loss(cls, cls_labels)
#####################


        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * cc_loss
        else:
            loss = 1.0 * cc_loss + 0.1 * disep_loss


        avg_meter.add({'cc_loss': cc_loss.item(), 'disep_loss': disep_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            cam_label = cam_label.cpu().numpy().astype(np.int16)
            #pred_cam = pred_cam.cpu().numpy().astype(np.int16)
            logging.info(
                "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cc_loss: %.4f; disep_loss: %.4f" % (   #; cp_loss: %.4f
                    n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cc_loss'), avg_meter.pop('disep_loss'))) #, avg_meter.pop('cp_loss')


            grid_imgs_A, grid_cam_A = imutils.tensorboard_image(imgs=inputs_A.clone(), cam=valid_cam)
            grid_imgs_B, grid_cam_B = imutils.tensorboard_image(imgs=inputs_B.clone(), cam=valid_cam)

###
            grid_pred_cam = imutils.tensorboard_label(labels=cam_label)


            writer.add_image("train/images_A"+str(img_name), grid_imgs_A, global_step=n_iter)
            writer.add_image("train/images_B"+str(img_name), grid_imgs_B, global_step=n_iter)
            writer.add_image("cam/valid_cams_A"+str(img_name), grid_cam_A, global_step=n_iter)
            writer.add_image("cam/valid_cams_B"+str(img_name), grid_cam_B, global_step=n_iter)
            writer.add_image("train/preds_cam", grid_pred_cam, global_step=n_iter)

            writer.add_scalars('train/cc_loss', {"cc_loss": cc_loss.item()},
                               global_step=n_iter)
            writer.add_scalars('train/disep_loss', {"disep_loss": disep_loss.item()},
                               global_step=n_iter)


        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "transwcd_iter_%d.pth" % (n_iter + 1))
            logging.info('CD Validating...')
            torch.save(transwcd.state_dict(), ckpt_name)
            cam_score, labels = validate(model=transwcd, data_loader=val_loader, cfg=cfg)  # _ 为 labels

            if cam_score['f1'][1] > best_F1:
                best_F1 = cam_score['f1'][1]
                best_iter = n_iter + 1
            logging.info("cams score: %s, \n[best_iter]: %s", cam_score, best_iter)

            if cam_score['f1'][1] > best_F1:
                best_F1 = cam_score['f1'][1]
                best_iter = n_iter + 1

    return True

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)


    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp + '.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg)


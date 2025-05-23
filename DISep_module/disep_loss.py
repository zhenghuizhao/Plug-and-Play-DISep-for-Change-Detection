import torch
import torch.nn as nn
import torch.nn.functional as F


class DISEP_Loss(nn.Module):
    def __init__(self, margin_uc_c=0.05, margin_c_c=0.0, p=1.0):  #margin_uc_c=0.05
        super(DISEP_Loss, self).__init__()
        self.margin_uc_c = margin_uc_c
        self.margin_c_c = margin_c_c
        self.p = p

    def pixel_to_center_loss(self, protos, feat_inst_dict, mask_inst_dict, cls_label):
        device = torch.device('cuda')
        loss_pixel = torch.zeros(1, device=device)
        reg_hard = torch.zeros(1, device=device)
        inst_num = 0.0
        hard_num = 0.0

        for img_idx, inst_feats in feat_inst_dict.items():
            # print('img_idx:', img_idx)
            for label, feat_inst in inst_feats.items():
                inst_mask = mask_inst_dict[img_idx][label]
                inst_size = torch.sum(inst_mask)  # Calculate instance size
                feat_center = protos[img_idx][label]

                if cls_label[img_idx] == 1.0:
                    loss_pixel += torch.norm(feat_center[:, None, None] - feat_inst, p=2) / inst_size
                    inst_num += 1
                elif cls_label[img_idx] == 0.0:
                    hard_size = (feat_inst.size(-1)**2)
                    reg_hard += torch.norm(feat_center[:, None, None] - feat_inst, p=2) / hard_size
                    hard_num += 1

        if inst_num != 0:
            # print('loss_pixel:', loss_pixel)
            # print('inst:', inst_num)
            loss_pixel = loss_pixel / inst_num
        else:
            loss_pixel = torch.tensor(loss_pixel)
        if hard_num != 0:
            reg_hard = reg_hard / hard_num
        else:
            loss_pixel = torch.tensor(reg_hard)

        # print('inst_num:', inst_num)
        # print('hard_num:', hard_num)
        # print('loss_pixel:', loss_pixel)
        # print('reg_hard:', reg_hard)

        return loss_pixel   # + 0.001 * reg_hard


    def forward(self, centers, feat_inst_dict, mask_inst_dict, cls_label):    # , feat_mask_c, feat_mask_uc, feat_mask_c_cent, feat_mask_uc_cent


        loss_pixel = self.pixel_to_center_loss(centers, feat_inst_dict, mask_inst_dict, cls_label)


        return loss_pixel




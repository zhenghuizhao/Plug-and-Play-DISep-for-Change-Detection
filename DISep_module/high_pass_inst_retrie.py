import cv2
import numpy as np
import torch
import torch.nn.functional as F

def hp_inst_retrie(cam, feat, cls_label, img_box=None, ignore_mid=True, cfg=None):
    b, c, h, w = cam.shape


    cam_value, _candidate = cam.max(dim=1, keepdim=False)
    _candidate += 1

    # uc is unchanged-in-changed; c is changed-in-changed; u is unchanged-in-unchanged
    candidate_uc = torch.where(cam_value >= cfg.mic.lowpass_score, 0, 1)
    candidate_c = torch.where(cam_value <= cfg.mic.highpass_score, 0, 1)


    mask_c = torch.tensor(candidate_c.cpu().numpy() == 1, dtype=torch.float32, device=candidate_c.device)
    mask_uc = torch.tensor(candidate_uc.cpu().numpy() == 1, dtype=torch.float32, device=candidate_uc.device)
    instance_masks_dict = {}
    inst_cent = {}
    feat_inst_dict = {}

    hp_cam = mask_c * cam


    for i in range(b):
        inst_masks = {}
        if cls_label[i] == 1.0:
            num_labels, labels, _, _ = cv2.connectedComponentsWithStats((mask_c[i].cpu().numpy() == 1).astype(np.uint8), connectivity=8)
            for label in range(1, num_labels):
                inst_mask = torch.tensor(labels == label, dtype=torch.float32, device=mask_c.device)
                inst_masks[f"c{label}"] = inst_mask
            inst_masks["uc"] = mask_uc[i]
        else:
            inst_masks["uc"] = cam_value[i] + 1e-9

        instance_masks_dict[i] = inst_masks


        inst_cent[i] = {}
        feat_inst_dict[i] = {}
        for label, inst_mask in inst_masks.items():
            feat_inst = feat[i] * inst_mask.unsqueeze(0).expand_as(feat[i])
            feat_cent = torch.mean(feat_inst, dim=(1, 2))
            if cls_label[i] == 0:
                feat_cent = torch.mean(feat[i], dim=(1, 2))
            inst_cent[i][label] = feat_cent
            feat_inst_dict[i][label] = feat_inst

    ### image-based centroid
        # mask_c = mask_c.unsqueeze(1)
        # mask_uc = mask_uc.unsqueeze(1)
        # feat_c = feat * mask_c
        # feat_uc = feat * mask_uc
        # # print('feat_c:', feat_mask_c.size())
        # # print('mask_c:', mask_c.size())
        # feat_mask_c_cent = torch.sum(feat_mask_c, dim=(2, 3)) / (torch.sum(mask_c, dim=(2, 3)) + 1e-9)
        # feat_mask_uc_cent = torch.sum(feat_mask_uc, dim=(2, 3)) / (torch.sum(mask_uc, dim=(2, 3)) + 1e-9)
    ###

    return hp_cam, inst_cent, feat_inst_dict, instance_masks_dict

import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class MLPHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    # in_channels = 128
    def __init__(self, feature_strides=[2, 4, 8, 16], in_channels=[32, 64, 128, 256], embedding_dim=64, in_index=[0, 1, 2, 3], num_classes=2, **kwargs):
        super(MLPHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.in_index = in_index
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim


        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)


        self.dropout = nn.Dropout2d(0.1)



        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )


        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # final predction head


    def forward(self, x):
        c1, c2, c3, c4 = x


        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        # Stage 4: x1/32 scale
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        ##### 变化检测增加的部分  做融合
        _c4_up = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])

        ##### 这个操作增加 F.interpolate(_c4, scale_factor=2, mode="bilinear")会使得模型更好吗？
        #print('_c4', _c4.size())
        #print('_c3_1', _c3_1.size())
        _c3_up = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2_up = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])


        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1], dim=1))   # logit

        x = self.dropout(_c)   #_c4
        #print('_c:', _c.size())    #_c: torch.Size([4, 256, 64, 64])
        #print('_c4_up:', _c4_up.size())    #_c4_up: torch.Size([4, 256, 64, 64])
        cd = self.linear_pred(x)

        return cd





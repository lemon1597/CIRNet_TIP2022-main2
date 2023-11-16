import torch
import torch.nn as nn
import torch.nn.functional as F
from module.BaseBlock import BaseConv2d, ChannelAttention


class RorD_Decoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RorD_Decoder, self).__init__()
        self.conv1 = BaseConv2d(in_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = BaseConv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, fea_before, fea_vgg):
        """
        Args:
            fea_before: previous decoder feature
            fea_vgg: previous encoder feature
        
        Returns:
            fea_out: the fused decoder feature

        """
        fea_mix = self.conv1(torch.cat((fea_before, fea_vgg), dim=1))
        fea_out = self.conv2(fea_mix)

        return fea_out


class IGF(nn.Module):
    """
    The implementation of the importance gated fusion (IGF) unit.
    """
    def __init__(self, fea_before_channels, fea_rd_channels, out_channels, up=True):
        super(IGF, self).__init__()
        self.up = up
        self.conv1 = BaseConv2d(fea_rd_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(fea_before_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_reduce = BaseConv2d(out_channels * 2, out_channels, kernel_size=1)
        self.ca = ChannelAttention(out_channels)
        self.conv_k = BaseConv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv3 = BaseConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv4 = BaseConv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, fea_before, fea_r, fea_d):
        """
        Args:
            fea_before: previous IGF feature
            fea_r: the fused rgb decoder feature
            fea_d: the fused depth decoder feature
        
        Returns:
            fea_out: the IGF output feature

        """
        fea_mix = self.conv1(torch.cat((fea_r, fea_d), dim=1)) # (B, 512, 7, 7)-->(B, 1024, 7, 7) --> (B, 512, 7, 7)
        fea_before_conv = self.conv2(fea_before)  # (B, 512, 7, 7) --> (B, 512, 7, 7)

        fea_cat_reduce = self.conv_reduce(torch.cat((fea_before_conv, fea_mix), dim=1)) # (B, 512, 7, 7) --> (B, 1024, 7, 7) --> (B, 512, 7, 7)
        fea_cat_reduce_ca = fea_cat_reduce.mul(self.ca(fea_cat_reduce)) + fea_cat_reduce # (B, 512, 7, 7) --> (B, 512, 7, 7)
        p_block = torch.sigmoid(self.conv_k(fea_cat_reduce_ca)) # (B, 512, 7, 7) --> (B, 512, 7, 7)
        one_block = torch.ones_like(p_block) # (B, 512, 7, 7)

        fea_out = fea_before_conv * (one_block - p_block) + fea_mix * p_block # (B, 512, 7, 7)
        fea_out = self.relu(self.bn(fea_out))
        fea_out = self.conv3(fea_out)
        fea_out = self.conv4(fea_out)
        if self.up:
            fea_out = F.interpolate(fea_out, scale_factor=2, mode='bilinear', align_corners=True) # (B, 512, 7, 7) -- > (B, 512, 14, 14)
            
        return fea_out


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        res_channels = [64, 256, 512, 1024, 2048]
        channels = [64, 128, 256, 512, 512]

        self.r1 = RorD_Decoder(channels[4], channels[3])
        self.r2 = RorD_Decoder(channels[3], channels[2])
        self.r3 = RorD_Decoder(channels[2], channels[1])
        self.r4 = RorD_Decoder(channels[1], channels[0])
        self.r5 = RorD_Decoder(channels[0], 3)

        self.d1 = RorD_Decoder(channels[4], channels[3])
        self.d2 = RorD_Decoder(channels[3], channels[2])
        self.d3 = RorD_Decoder(channels[2], channels[1])
        self.d4 = RorD_Decoder(channels[1], channels[0])
        self.d5 = RorD_Decoder(channels[0], 3)

        self.rd1 = IGF(channels[4], channels[3], channels[3])
        self.rd2 = IGF(channels[3], channels[2], channels[2])
        self.rd3 = IGF(channels[2], channels[1], channels[1])
        self.rd4 = IGF(channels[1], channels[0], channels[0])
        self.rd5 = IGF(channels[0], 3, 3)

        self.conv_r_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.conv_d_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.conv_rgbd_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, rgb_list, depth_list, rgbd):
        """
        Args:
            rgb_list: the list of rgb encoder features
            depth_list: the list of depth encoder features
            rgbd: the refine rgbd feature by cmWR unit
        
        Returns:
            the saliency map of rgb, depth, rgbd stream

        """
        # rgb decoder stream
        rgb_block5 = self.r1(rgb_list[5], rgb_list[4])  # (B, 512, 7, 7),(B, 512, 7, 7)-->(B, 1023, 7, 7)--> [B, 512, 7, 7]
        rgb_block5_up = F.interpolate(rgb_block5, scale_factor=2, mode='bilinear') # 双线性插值，尺度翻2倍，(B, 512, 7, 7) --> (B, 512, 14, 14)
        rgb_block4 = self.r2(rgb_block5_up, rgb_list[3])  # (B, 512, 14, 14),(B, 512, 14, 14) --> (B, 1024, 14, 14) --> (B, 256, 14, 14)
        rgb_block4_up = F.interpolate(rgb_block4, scale_factor=2, mode='bilinear') # (B, 256, 14, 14) --> (B, 256, 28, 28)
        rgb_block3 = self.r3(rgb_block4_up, rgb_list[2])  # (B, 256, 28, 28),(B, 256, 28, 28) --> (B, 512, 28, 28) --> (B, 128, 28, 28)
        rgb_block3_up = F.interpolate(rgb_block3, scale_factor=2, mode='bilinear') # (B, 128, 28, 28) --> (B, 128, 56, 56)
        rgb_block2 = self.r4(rgb_block3_up, rgb_list[1])  # (B, 128, 56, 56),(B, 128, 56, 56) --> (B, 256, 56, 56) --> (B, 64, 56, 56)
        rgb_block2_up = F.interpolate(rgb_block2, scale_factor=2, mode='bilinear') # (B, 64, 56, 56) --> (B, 64, 112, 112)
        rgb_block1 = self.r5(rgb_block2_up, rgb_list[0])  # (B, 64, 112, 112), (B, 64, 112, 112) --> (B, 128, 112, 112) --> (B, 3, 112, 112)
        rgb_block1_up = F.interpolate(rgb_block1, scale_factor=2, mode='bilinear') # (B, 3, 112, 112) --> (B, 3, 224, 224)
        rgb_map = self.conv_r_map(rgb_block1_up)  # (B, 3, 224, 224) --> (B, 1, 224, 224)

        # depth decoder stream
        depth_block5 = self.d1(depth_list[5], depth_list[4])
        depth_block5_up = F.interpolate(depth_block5, scale_factor=2, mode='bilinear')
        depth_block4 = self.d2(depth_block5_up, depth_list[3])
        depth_block4_up = F.interpolate(depth_block4, scale_factor=2, mode='bilinear')
        depth_block3 = self.d3(depth_block4_up, depth_list[2])
        depth_block3_up = F.interpolate(depth_block3, scale_factor=2, mode='bilinear')
        depth_block2 = self.d4(depth_block3_up, depth_list[1])
        depth_block2_up = F.interpolate(depth_block2, scale_factor=2, mode='bilinear')
        depth_block1 = self.d5(depth_block2_up, depth_list[0])
        depth_block1_up = F.interpolate(depth_block1, scale_factor=2, mode='bilinear')
        depth_map = self.conv_d_map(depth_block1_up)

        # rgbd decoder stream
        rgbd_block5 = self.rd1(rgbd, rgb_block5, depth_block5) # [(B, 512, 7, 7),(B, 512, 7, 7),(B, 512, 7, 7)] --> (B, 512, 14, 14)
        rgbd_block4 = self.rd2(rgbd_block5, rgb_block4, depth_block4) # [(B, 512, 14, 14),(B, 512, 14, 14),(B, 512, 14, 14)] --> (B, 256, 28, 28)
        rgbd_block3 = self.rd3(rgbd_block4, rgb_block3, depth_block3) # [(B, 256, 28, 28),(B, 256, 28, 28),(B, 256, 28, 28)] --> (B, 128, 56, 56)
        rgbd_block2 = self.rd4(rgbd_block3, rgb_block2, depth_block2) # [(B, 128, 56, 56),(B, 128, 56, 56),(B, 128, 56, 56)] --> (B, 64, 112, 112)
        rgbd_block1 = self.rd5(rgbd_block2, rgb_block1, depth_block1) # [(B, 3, 112, 112),(B, 3, 112, 112),(B, 3, 112, 112)] --> (B, 3, 224, 224)
        rgbd_map = self.conv_rgbd_map(rgbd_block1) # (B, 3, 224, 224)

        return rgb_map, depth_map, rgbd_map




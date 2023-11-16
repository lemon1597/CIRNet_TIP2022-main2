import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.ResNet import Backbone_ResNet50
from module.cmWR import cmWR
from module.BaseBlock import BaseConv2d, SpatialAttention, ChannelAttention
from module.Decoder import Decoder
from module.CGA import CGA


class CIRNet_R50(nn.Module):
    """
    The implementation of "CIR-Net: Cross-Modality Interaction and Refinement for RGB-D Salient Object Detection"
    """
    def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm2d):
        # 
        super(CIRNet_R50, self).__init__()
        (
            self.rgb_block1, # (B, 3, H, W) --> (B, 64, H/2, W/2)
            self.rgb_block2, # (B, 64, H, W) --> (B, 256, H/2, W/2)
            self.rgb_block3, # (B, 256, H/2, W/2) --> (B, 512, H/4, W/4)
            self.rgb_block4, # (B, 512, H/4, W/4) --> (B, 1024, H/8, W/8)
            self.rgb_block5, # (B, 1024, H/8, W/8) --> (B, 2048, H/16, W/16)
        ) = Backbone_ResNet50(pretrained=True)

        (
            self.depth_block1,
            self.depth_block2,
            self.depth_block3,
            self.depth_block4,
            self.depth_block5,
        ) = Backbone_ResNet50(pretrained=True)


        res_channels = [64, 256, 512, 1024, 2048] # resnet网络中的每个layer的输入通道数（或者，每前面一层最后一个块的输出通道数）
        #
        channels = [64, 128, 256, 512, 512] # resnet网络中每个layer的第一个块的输出通道数

        # layer 1
        self.re1_r = BaseConv2d(res_channels[0], channels[0], kernel_size=1)
        self.re1_d = BaseConv2d(res_channels[0], channels[0], kernel_size=1)
        # self.re1_r_cga = CGA(channels[0])
        # self.re1_d_cga = CGA(channels[0])

        # layer 2
        # self.re2_r = BaseConv2d(res_channels[1], channels[1], kernel_size=1)
        # self.re2_d = BaseConv2d(res_channels[1], channels[1], kernel_size=1)
        self.re2_r_cga = CGA(dim=129,embed_dim=129,in_chans=res_channels[1],out_chans=channels[1])
        self.re2_d_cga = CGA(dim=129,embed_dim=129,in_chans=res_channels[1],out_chans=channels[1])

        # layer 3
        # self.re3_r = BaseConv2d(res_channels[2], channels[2], kernel_size=1)
        # self.re3_d = BaseConv2d(res_channels[2], channels[2], kernel_size=1)
        self.re3_r_cga = CGA(dim=258,embed_dim=258,in_chans=res_channels[2],out_chans=channels[2])
        self.re3_d_cga = CGA(dim=258,embed_dim=258,in_chans=res_channels[2],out_chans=channels[2])
         
        self.conv1 = BaseConv2d(2 * channels[2], channels[2], kernel_size=1)
        self.sa1 = SpatialAttention(kernel_size=7)

        # layer 4
        # self.re4_r = BaseConv2d(res_channels[3], channels[3], kernel_size=1)
        # self.re4_d = BaseConv2d(res_channels[3], channels[3], kernel_size=1)
        self.re4_r_cga = CGA(dim=516,embed_dim=516,in_chans=res_channels[3],out_chans=channels[3])
        self.re4_d_cga = CGA(dim=516,embed_dim=516,in_chans=res_channels[3],out_chans=channels[3])

        self.conv2 = BaseConv2d(2 * channels[3], channels[3], kernel_size=1)
        self.sa2 = SpatialAttention(kernel_size=7)

        # layer 5
        # self.re5_r = BaseConv2d(res_channels[4], channels[4], kernel_size=1)
        # self.re5_d = BaseConv2d(res_channels[4], channels[4], kernel_size=1)
        self.re5_r_cga = CGA(dim=1032,embed_dim=1032,in_chans=res_channels[4],out_chans=channels[4])
        self.re5_d_cga = CGA(dim=1032,embed_dim=1032,in_chans=res_channels[4],out_chans=channels[4])

        self.conv3 = BaseConv2d(2 * channels[4], channels[4], kernel_size=1)

        # self-modality attention refinement 
        self.ca_rgb = ChannelAttention(channels[4])
        self.ca_depth = ChannelAttention(channels[4])
        self.ca_rgbd = ChannelAttention(channels[4])

        self.sa_rgb = SpatialAttention(kernel_size=7)
        self.sa_depth = SpatialAttention(kernel_size=7)
        self.sa_rgbd = SpatialAttention(kernel_size=7)

        # cross-modality weighting refinement
        self.cmWR = cmWR(channels[4], squeeze_ratio=1)

        self.conv_rgb = BaseConv2d(channels[4], channels[4], kernel_size = 3, padding=1)
        self.conv_depth = BaseConv2d(channels[4], channels[4], kernel_size = 3, padding=1)
        self.conv_rgbd = BaseConv2d(channels[4], channels[4], kernel_size = 3, padding=1)

        self.decoder = Decoder()


    def forward(self, rgb, depth):
        decoder_rgb_list = []
        decoder_depth_list = []
        depth = torch.cat((depth, depth, depth), dim=1)

        # encoder layer 1
        conv1_res_r = self.rgb_block1(rgb) # (B, 3, 224, 224) --> (B, 64, 112, 112)
        conv1_res_d = self.depth_block1(depth)
        # conv1_r = self.re1_r(conv1_res_r) # (B, 64, 112, 112) --> (B, 64, 112, 112)
        # conv1_d = self.re1_d(conv1_res_d)
        # dfs = conv1_res_r.shape
        # dfss = conv1_res_r.shape[2]
        # print(dfs)
        # print(conv1_res_r.shape[2])

        # CGA
        # conv1_r_cga = self.re1_r_cga(conv1_res_r,conv1_res_r.shape[2],conv1_res_r.shape[3])  # input = (64, 112, 112), output = (64, 112, 112)
        # conv1_d_cga = self.re1_d_cga(conv1_res_d,conv1_res_d.shape[2],conv1_res_d.shape[3])  # input = (64, 112, 112), output = (64, 112, 112)

        decoder_rgb_list.append(conv1_res_r)
        decoder_depth_list.append(conv1_res_d)

        # encoder layer 2
        conv2_res_r = self.rgb_block2(conv1_res_r) # (B, 64, 112, 112) --> (B, 64, 56, 56) --> (B, 256, 56, 56)
        conv2_res_d = self.depth_block2(conv1_res_d)
        # conv2_r = self.re2_r(conv2_res_r) # (B, 256, 56, 56) --> (B, 128, 56, 56)
        # conv2_d = self.re2_d(conv2_res_d)

        # CGA
        conv2_r_cga = self.re2_r_cga(conv2_res_r)
        conv2_d_cga = self.re2_d_cga(conv2_res_d)

        decoder_rgb_list.append(conv2_r_cga)
        decoder_depth_list.append(conv2_d_cga)

        # encoder layer 3
        conv3_res_r = self.rgb_block3(conv2_res_r) # (B, 256, 56, 56) --> (B, 128, 28, 28) --> (B, 512, 28, 28)
        conv3_res_d = self.depth_block3(conv2_res_d)
        # conv3_r = self.re3_r(conv3_res_r) # (B, 512, 28, 28) --> (B, 256, 28, 28)
        # conv3_d = self.re3_d(conv3_res_d)

        # CGA
        conv3_r_cga = self.re3_r_cga(conv3_res_r)
        conv3_d_cga = self.re3_d_cga(conv3_res_d)

        # progressive attention guided integration unit
        conv3_rgbd = self.conv1(torch.cat((conv3_r_cga, conv3_d_cga), dim=1)) # (B, 256+256, 28, 28)==(B, 512, 28, 28) --> (B, 256, 28, 28)
        conv3_rgbd = F.interpolate(conv3_rgbd, scale_factor=1/2, mode='bilinear', align_corners=True) # 双线性插值，缩小特征图尺寸，(B, 256, 28, 28) --> (B, 256, 14, 14)
        conv3_rgbd_map = self.sa1(conv3_rgbd) # (B, 256, 14, 14) --> (B, 1, 14, 14)
        decoder_rgb_list.append(conv3_r_cga)
        decoder_depth_list.append(conv3_d_cga)

        # encoder layer 4
        conv4_res_r = self.rgb_block4(conv3_res_r) # (B, 512, 28, 28) --> (B, 256, 14, 14) --> (B, 1024, 14, 14)
        conv4_res_d = self.depth_block4(conv3_res_d)
        # conv4_r = self.re4_r(conv4_res_r) # (B, 1024, 14, 14) --> (B, 512, 14, 14)
        # conv4_d = self.re4_d(conv4_res_d)

        # CGA
        conv4_r_cga = self.re4_r_cga(conv4_res_r)
        conv4_d_cga = self.re4_d_cga(conv4_res_d)

        conv4_rgbd = self.conv2(torch.cat((conv4_r_cga, conv4_d_cga), dim=1)) # (B, 512+512, 14, 14)==(B, 1024, 14, 14) --> (B, 512, 14, 14)
        conv4_rgbd = conv4_rgbd * conv3_rgbd_map + conv4_rgbd # (B, 512, 14, 14) * (B, 1, 14, 14) + (B, 512, 14, 14) == (B, 512, 14, 14)
        conv4_rgbd = F.interpolate(conv4_rgbd, scale_factor=1/2, mode='bilinear', align_corners=True) # (B, 512, 7, 7)
        conv4_rgbd_map = self.sa2(conv4_rgbd) # (B, 512, 7, 7) -->  (B, 1, 7, 7)
        decoder_rgb_list.append(conv4_r_cga)
        decoder_depth_list.append(conv4_d_cga)

        # encoder layer 5
        conv5_res_r = self.rgb_block5(conv4_res_r) # (B, 1024, 14, 14) --> (B, 512, 7, 7) --> (B, 2048, 7, 7)
        conv5_res_d = self.depth_block5(conv4_res_d)
        # conv5_r = self.re5_r(conv5_res_r) # (B, 2048, 7, 7) --> (B, 512, 7, 7)
        # conv5_d = self.re5_d(conv5_res_d)

        # CGA
        conv5_r_cga = self.re5_r_cga(conv5_res_r)
        conv5_d_cga = self.re5_d_cga(conv5_res_d)

        conv5_rgbd = self.conv3(torch.cat((conv5_r_cga, conv5_d_cga), dim=1)) # (B, 512 + 512, 7, 7) == (B, 1024, 7, 7) --> (B, 512, 7, 7)
        conv5_rgbd = conv5_rgbd * conv4_rgbd_map + conv5_rgbd # (B, 512, 7, 7)*(B, 1, 7, 7) + (B, 512, 7, 7) == (B, 512, 7, 7)
        decoder_rgb_list.append(conv5_r_cga) # [(B, 64, 112, 112), (B, 128, 56, 56), (B, 256, 28, 28), (B, 512, 14, 14), (B, 512, 7, 7)]
        decoder_depth_list.append(conv5_d_cga)

        # self-modality attention refinement
        B, C, H, W = conv5_r_cga.size() # (B, 512, 7, 7)
        P = H * W # 49

        rgb_SA = self.sa_rgb(conv5_r_cga).view(B, -1, P)    # (B, 1, 49)        # B * 1 * H * W
        depth_SA = self.sa_depth(conv5_d_cga).view(B, -1, P)
        rgbd_SA = self.sa_rgbd(conv5_rgbd).view(B, -1, P)

        rgb_CA = self.ca_rgb(conv5_r_cga).view(B, C, -1) # (B, 512, 1)          # B * C * 1 * 1
        depth_CA = self.ca_depth(conv5_d_cga).view(B, C, -1)
        rgbd_CA = self.ca_rgbd(conv5_rgbd).view(B, C, -1)

        rgb_M = torch.bmm(rgb_CA, rgb_SA).view(B, C, H, W) # (B, 512, 7, 7)
        depth_M = torch.bmm(depth_CA, depth_SA).view(B, C, H, W)
        rgbd_M = torch.bmm(rgbd_CA, rgbd_SA).view(B, C, H, W)

        rgb_smAR = conv5_r_cga *  rgb_M + conv5_r_cga  # (B, 512, 7, 7)
        depth_smAR = conv5_d_cga * depth_M + conv5_d_cga
        rgbd_smAR = conv5_rgbd * rgbd_M + conv5_rgbd 

        rgb_smAR = self.conv_rgb(rgb_smAR) # (B, 512, 7, 7)
        depth_smAR = self.conv_depth(depth_smAR)
        rgbd_smAR = self.conv_rgbd(rgbd_smAR)

        # cross-modality weighting refinement
        rgb_cmWR, depth_cmWR, rgbd_cmWR = self.cmWR(rgb_smAR, depth_smAR, rgbd_smAR)  # (B, 512, 7, 7)
        
        decoder_rgb_list.append(rgb_cmWR) # [(B, 64, 112, 112), (B, 128, 56, 56), (B, 256, 28, 28), (B, 512, 14, 14), (B, 512, 7, 7)，(B, 512, 7, 7)]
        decoder_depth_list.append(depth_cmWR)
        
        # decoder
        rgb_map, depth_map, rgbd_map = self.decoder(decoder_rgb_list, decoder_depth_list, rgbd_cmWR)


        return rgb_map, depth_map, rgbd_map
# if __name__ == "__main__":
#     model = CIRNet_R50()
#     print(model)
#     input1 = torch.rand([32, 3, 352, 352])
#     input2 = torch.rand([32,1,352,352])
#     print(model(input1, input2))
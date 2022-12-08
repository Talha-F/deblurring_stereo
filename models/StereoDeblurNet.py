#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
"""
## Multi-Scale-Stage Network
## Code is based on Multi-Stage Progressive Image Restoration(MPRNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from utils.network_utils import *
from models.submodules import *

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

##########################################################################
## Residual Block
class ResBlock(nn.Module):
    def __init__(self, wf, kernel_size, reduction, bias, act):
        super(ResBlock, self).__init__()
        modules_body = []
        modules_body.append(conv(wf, wf, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(wf, wf, kernel_size, bias=bias))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, wf, scale, vscale, kernel_size, reduction, act, bias, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [ResBlock(wf,              kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [ResBlock(wf+scale,        kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [ResBlock(wf+scale+vscale, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(wf, scale)
        self.down23  = DownSample(wf+scale, vscale)

        if csff:
            self.csff_enc1 = nn.Conv2d(wf,              wf,              kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(wf+scale,        wf+scale,        kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(wf+scale+vscale, wf+scale+vscale, kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(wf,              wf,              kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(wf+scale,        wf+scale,        kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(wf+scale+vscale, wf+scale+vscale, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, wf, scale, vscale, kernel_size, reduction, act, bias):
        super(Decoder, self).__init__()

        self.decoder_level1 = [ResBlock(wf,              kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [ResBlock(wf+scale,        kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [ResBlock(wf+scale+vscale, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = ResBlock(wf,       kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = ResBlock(wf+scale, kernel_size, reduction, bias=bias, act=act)

        self.up32  = SkipUpSample(wf+scale, vscale)
        self.up21  = SkipUpSample(wf, scale)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class EDUp(nn.Module):
    def __init__(self):
        super(EDUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, elist, dlist):

        up_elist = []
        for feat in elist:
            up_elist.append(self.up(feat))

        up_dlist = []
        for feat in dlist:
            up_dlist.append(self.up(feat))

        return up_elist, up_dlist

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class ScaleUpSample(nn.Module):
    def __init__(self, in_channels):
        super(ScaleUpSample,self).__init__()
        self.scaleUp = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                     nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))
    def forward(self,feat):
        return self.scaleUp(feat)

#https://github.com/fangwei123456/PixelUnshuffle-pytorch
class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * (k*w) * (k*h)
        kdownscale_factor: k
        batchSize * c * (k*w) * (k*h) -> batchSize * (k*k*c) * w * h
        '''
        c = input.shape[1]

        kernel = torch.zeros(size=[self.downscale_factor * self.downscale_factor * c,
                                   1, self.downscale_factor, self.downscale_factor],
                             device=input.device)
        for y in range(self.downscale_factor):
            for x in range(self.downscale_factor):
                kernel[x + y * self.downscale_factor::self.downscale_factor*self.downscale_factor, 0, y, x] = 1
        return F.conv2d(input, kernel, stride=self.downscale_factor, groups=c)

class Tail_shuffle(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,bias):
        super(Tail_shuffle,self).__init__()
        self.tail = conv(in_channels, out_channels, kernel_size, bias=bias)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, feat):
        return self.pixelshuffle(self.tail(feat))

##########################################################################

class StereoDeblurNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, wf=32, scale=10, vscale=10, kernel_size=3, reduction=4, bias=False):
        super(StereoDeblurNet, self).__init__()
        # encoder
        act=nn.PReLU()
        self.pixel_unshuffle = PixelUnshuffle(2)
        self.ED_up = EDUp()
        #scale1
        self.shallow_feat1 = nn.Sequential(conv(12, wf, kernel_size, bias=bias), ResBlock(wf,kernel_size, reduction, bias=bias, act=act))
        self.E1_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=False)
        self.D1_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        self.tail1_1 = Tail_shuffle(wf, 12, kernel_size, bias=bias)
        ##########################################################################
        #scale2
        self.shallow_feat2 = nn.Sequential(conv(12, wf, kernel_size, bias=bias), ResBlock(wf,kernel_size, reduction, bias=bias, act=act))
        self.up_scale1_feat = ScaleUpSample(wf)
        self.fusion12 = conv(wf*2, wf, kernel_size, bias=bias)

        self.E2_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D2_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        self.E2_2 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D2_2 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        self.tail2_1 = Tail_shuffle(wf, 12, kernel_size, bias=bias)
        self.tail2_2 = Tail_shuffle(wf, 12, kernel_size, bias=bias)
        ################################################################################
        #scale3
        self.shallow_feat3 = nn.Sequential(conv(3, wf, kernel_size, bias=bias), ResBlock(wf,kernel_size, reduction, bias=bias, act=act))
        self.up_scale2_feat = ScaleUpSample(wf)
        self.fusion23 = conv(wf*2, wf, kernel_size, bias=bias)

        self.E3_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D3_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        self.E3_2 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D3_2 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        self.E3_3 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D3_3 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        self.tail3_1 = conv(wf, 3, kernel_size, bias=bias)
        self.tail3_2 = conv(wf, 3, kernel_size, bias=bias)
        self.tail3_3 = conv(wf, 128, kernel_size, bias=bias)
        
        ks = 3
        self.conv1_1 = conv(3, 32, kernel_size=ks, stride=1)
        self.conv1_2 = resnet_block(32, kernel_size=ks)
        self.conv1_3 = resnet_block(32, kernel_size=ks)
        self.conv1_4 = resnet_block(32, kernel_size=ks)

        self.conv2_1 = conv(32, 64, kernel_size=ks, stride=2)
        self.conv2_2 = resnet_block(64, kernel_size=ks)
        self.conv2_3 = resnet_block(64, kernel_size=ks)
        self.conv2_4 = resnet_block(64, kernel_size=ks)

        self.conv3_1 = conv(64, 128, kernel_size=ks, stride=2)
        self.conv3_2 = resnet_block(128, kernel_size=ks)
        self.conv3_3 = resnet_block(128, kernel_size=ks)
        self.conv3_4 = resnet_block(128, kernel_size=ks)

        dilation = [1,2,3,4]
        self.convd_1 = resnet_block(128, kernel_size=ks, dilation = [2, 1])
        self.convd_2 = resnet_block(128, kernel_size=ks, dilation = [3, 1])
        self.convd_3 = ms_dilate_block(356, kernel_size=ks, dilation = dilation)

        self.gatenet = gatenet()

        self.depth_sense_l = depth_sense(33, 32, kernel_size=ks)
        self.depth_sense_r = depth_sense(33, 32, kernel_size=ks)

        # decoder
        self.upconv3_i = conv(288, 128, kernel_size=ks,stride=1)
        self.upconv3_3 = resnet_block(128, kernel_size=ks)
        self.upconv3_2 = resnet_block(128, kernel_size=ks)
        self.upconv3_1 = resnet_block(128, kernel_size=ks)

        self.upconv2_u = upconv(128, 64)
        self.upconv2_i = conv(128, 64, kernel_size=ks,stride=1)
        self.upconv2_3 = resnet_block(64, kernel_size=ks)
        self.upconv2_2 = resnet_block(64, kernel_size=ks)
        self.upconv2_1 = resnet_block(64, kernel_size=ks)

        self.upconv1_u = upconv(64, 32)
        self.upconv1_i = conv(64, 32, kernel_size=ks,stride=1)
        self.upconv1_3 = resnet_block(32, kernel_size=ks)
        self.upconv1_2 = resnet_block(32, kernel_size=ks)
        self.upconv1_1 = resnet_block(32, kernel_size=ks)

        self.img_prd = conv(32, 3, kernel_size=ks, stride=1)


    def forward(self, imgs, disps_bi, disp_feature):
        img_left  = imgs[:,:3]
        img_right = imgs[:,3:]
        img_left = img_left.float()
        img_right = img_right.float()

        disp_left  = disps_bi[:, 0]
        disp_right = disps_bi[:, 1]
        
        interpolation = nn.Upsample(scale_factor=0.5, mode = 'bilinear',align_corners=True)
        s2_blur_left = interpolation(img_left)
        s2_blur_right= interpolation(img_right)
         ##-------------------------------------------
        ##-------------- Scale 1---------------------
        ##-------------------------------------------
        s1_blur_ps_left = self.pixel_unshuffle(s2_blur_left)
        s1_blur_ps_right= self.pixel_unshuffle(s2_blur_right)

        shfeat1_left = self.shallow_feat1(s1_blur_ps_left)
        shfeat1_right= self.shallow_feat1(s1_blur_ps_right)

        e1_1f_left = self.E1_1(shfeat1_left)
        e1_1f_right= self.E1_1(shfeat1_right)

        d1_1f_left = self.D1_1(e1_1f_left)
        d1_1f_right= self.D1_1(e1_1f_right)

        # res1_1_left = self.tail1_1(d1_1f_left) + s2_blur_left
        # res1_1_right= self.tail1_1(d1_1f_right) + s2_blur_right

        ##-------------------------------------------
        ##-------------- Scale 2---------------------
        ##-------------------------------------------

        s2_blur_ps_left = self.pixel_unshuffle(img_left)
        s2_blur_ps_right= self.pixel_unshuffle(img_right)

        shfeat2_left = self.shallow_feat2(s2_blur_ps_left)
        shfeat2_right= self.shallow_feat2(s2_blur_ps_right)

        s1_sol_feat_left = self.up_scale1_feat(d1_1f_left[0])
        s1_sol_feat_right= self.up_scale1_feat(d1_1f_right[0])
        fusion12_left = self.fusion12(torch.cat([shfeat2_left, s1_sol_feat_left], dim=1))
        fusion12_right= self.fusion12(torch.cat([shfeat2_right,s1_sol_feat_right], dim=1))

        e_list_left, d_list_left = self.ED_up(e1_1f_left, d1_1f_left)
        e_list_right,d_list_right= self.ED_up(e1_1f_right,d1_1f_right)
        e2_1f_left = self.E2_1(fusion12_left, e_list_left, d_list_left)
        e2_1f_right= self.E2_1(fusion12_right,e_list_right, d_list_right)
        d2_1f_left = self.D2_1(e2_1f_left)
        d2_1f_right= self.D2_1(e2_1f_right)

        e2_2f_left = self.E2_2(d2_1f_left[0], e2_1f_left, d2_1f_left)
        e2_2f_right= self.E2_2(d2_1f_right[0],e2_1f_right,d2_1f_right)
        d2_2f_left = self.D2_2(e2_2f_left)
        d2_2f_right= self.D2_2(e2_2f_right)

        # res2_1_left = self.tail2_1(d2_1f_left[0]) + img_left
        # res2_1_right= self.tail2_1(d2_1f_right[0])+ img_right

        ##-------------------------------------------
        ##-------------- Scale 3---------------------
        ##-------------------------------------------
        shfeat3_left = self.shallow_feat3(img_left)
        shfeat3_right= self.shallow_feat3(img_right)
        s2_sol_feat_left = self.up_scale2_feat(d2_2f_left[0])
        s2_sol_feat_right= self.up_scale2_feat(d2_2f_right[0])
        fusion23_left = self.fusion23(torch.cat([shfeat3_left, s2_sol_feat_left], dim=1))
        fusion23_right= self.fusion23(torch.cat([shfeat3_right,s2_sol_feat_right], dim=1))

        e_list_left, d_list_left = self.ED_up(e2_2f_left, d2_2f_left)
        e_list_right,d_list_right= self.ED_up(e2_2f_right,d2_2f_right)
        e3_1f_left = self.E3_1(fusion23_left, e_list_left, d_list_left)
        e3_1f_right= self.E3_1(fusion23_right,e_list_right, d_list_right)
        d3_1f_left = self.D3_1(e3_1f_left)
        d3_1f_right= self.D3_1(e3_1f_right)

        e3_2f_left = self.E3_2(d3_1f_left[0], e3_1f_left, d3_1f_left)
        e3_2f_right= self.E3_2(d3_1f_right[0],e3_1f_right,d3_1f_right)
        d3_2f_left = self.D3_2(e3_2f_left)
        d3_2f_right= self.D3_2(e3_2f_right)

        e3_3f_left = self.E3_3(d3_2f_left[0], e3_2f_left, d3_2f_left)
        e3_3f_right= self.E3_3(d3_2f_right[0],e3_2f_right,d3_2f_right)
        d3_3f_left = self.D3_3(e3_3f_left)
        d3_3f_right= self.D3_3(e3_3f_right)

        # res3_1_left = self.tail3_1(d3_1f_left[0]) + img_left
        # res3_1_right= self.tail3_1(d3_1f_right[0])+ img_right
        # res3_2_left = self.tail3_2(d3_2f_left[0]) + img_left
        # res3_2_right= self.tail3_2(d3_2f_right[0])+ img_right
        
        res3_3_left = self.tail3_3(d3_3f_left[0]) 
        res3_3_right= self.tail3_3(d3_3f_right[0])
        # res3_3_left = res3_3_left + img_left
        # res3_3_right= res3_3_right+ img_right
        print('res3_3_left.shape', res3_3_left.shape)
        res3_3_left = nn.functional.adaptive_avg_pool2d(res3_3_left, (64, 64))
        res3_3_right= nn.functional.adaptive_avg_pool2d(res3_3_right,(64, 64))

        print('res3_3_left_max.shape', res3_3_left.shape)
        
        b, c, h, w = res3_3_left.shape
        print('res3_3_left.shape', res3_3_left.shape)
        print('img_right.shape', img_right.shape)
        new_size = (disp_left*cfg.DATA.DIV_DISP)
        new_size = new_size[:, :img_right.shape[2], :img_right.shape[3]]
        print('new_size', new_size.shape)
        warp_img_left = disp_warp(img_right, -new_size, cuda=True)
        warp_img_right = disp_warp(img_left, new_size, cuda=True)
        print('warp_img_left.shape', warp_img_left.shape)
        print('warp_img_right.shape', warp_img_right.shape)
        diff_left = torch.sum(torch.abs(img_left - warp_img_left), 1).view(b,1,*warp_img_left.shape[-2:])
        diff_right = torch.sum(torch.abs(img_right - warp_img_right), 1).view(b,1,*warp_img_right.shape[-2:])
        diff_2_left = nn.functional.adaptive_avg_pool2d(diff_left, (h, w))
        diff_2_right = nn.functional.adaptive_avg_pool2d(diff_right, (h, w))

        disp_2_left = nn.functional.adaptive_avg_pool2d(disp_left, (h, w))
        disp_2_right = nn.functional.adaptive_avg_pool2d(disp_right, (h, w))

        disp_feature_2 = nn.functional.adaptive_avg_pool2d(disp_feature, (h, w))

        depth_aware_left = self.depth_sense_l(torch.cat([disp_feature_2, disp_2_left.view(b,1,h,w)], 1))
        depth_aware_right = self.depth_sense_r(torch.cat([disp_feature_2, disp_2_right.view(b,1,h,w)], 1))

        # the larger, the more accurate
        gate_left  = self.gatenet(diff_2_left)
        gate_right = self.gatenet(diff_2_right)

        warp_convd_left  = disp_warp(res3_3_right, -disp_2_left)
        warp_convd_right = disp_warp(res3_3_left, disp_2_right)

        # aggregate features
        agg_left  = res3_3_left * (1.0-gate_left) + warp_convd_left * gate_left.repeat(1,c,1,1)
        agg_right = res3_3_right * (1.0-gate_right) + warp_convd_right * gate_right.repeat(1,c,1,1)
        print('agg_left.shape', agg_left.shape)
        print('agg_right.shape', agg_right.shape)
        

        # decoder-left
        cat3_left = self.upconv3_i(torch.cat([res3_3_left, agg_left, depth_aware_left], 1))
        upconv3_left = self.upconv3_1(self.upconv3_2(self.upconv3_3(cat3_left)))                       # upconv3 feature

        upconv2_u_left = self.upconv2_u(upconv3_left)
        upconv2_u_left = upconv2_u_left[:, :, 0:res3_3_left.size()[2], 0:res3_3_left.size()[3]]
        print('upconv2_u_left.shape', upconv2_u_left.shape)
        print('res3_3_left.shape', res3_3_left.shape)
        cat2_left = self.upconv2_i(torch.cat([res3_3_left, upconv2_u_left],1))
        upconv2_left = self.upconv2_1(self.upconv2_2(self.upconv2_3(cat2_left)))                       # upconv2 feature
        upconv1_u_left = self.upconv1_u(upconv2_left)
        cat1_left = self.upconv1_i(torch.cat([res3_3_left, upconv1_u_left], 1))

        upconv1_left = self.upconv1_1(self.upconv1_2(self.upconv1_3(cat1_left)))                       # upconv1 feature
        img_prd_left = self.img_prd(upconv1_left) + img_left                                           # predict img

        # decoder-right
        cat3_right = self.upconv3_i(torch.cat([res3_3_right, agg_right, depth_aware_right], 1))
        upconv3_right = self.upconv3_1(self.upconv3_2(self.upconv3_3(cat3_right)))                     # upconv3 feature

        upconv2_u_right = self.upconv2_u(upconv3_right)
        upconv2_u_right = upconv2_u_right[:, :, 0:res3_3_right.size()[2], 0:res3_3_right.size()[3]]
        
        cat2_right = self.upconv2_i(torch.cat([res3_3_right, upconv2_u_right], 1))
        upconv2_right = self.upconv2_1(self.upconv2_2(self.upconv2_3(cat2_right)))                     # upconv2 feature
        upconv1_u_right = self.upconv1_u(upconv2_right)
        cat1_right = self.upconv1_i(torch.cat([res3_3_right, upconv1_u_right], 1))

        upconv1_right = self.upconv1_1(self.upconv1_2(self.upconv1_3(cat1_right)))                     # upconv1 feature
        img_prd_right = self.img_prd(upconv1_right) + img_right                                        # predict img

        imgs_prd = [img_prd_left, img_prd_right]

        diff = [diff_left, diff_right]
        gate = [gate_left, gate_right]

        return imgs_prd, diff, gate

        
from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import random

from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .gem import GeneralizedMeanPoolingP
from .metric import build_metric

__all__ = ['ResNetIBN', 'resnet_ibn50a', 'resnet_ibn101a']

class Waveblock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(0.3 * h)
            sx = random.randint(0, h-rh)
            mask = (x.new_ones(x.size()))*1.5
            mask[:, :, sx:sx+rh, :] = 1
            x = x * mask 
        return x

class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(ResNetIBN, self).__init__()
        
        self.pretrained = False
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNetIBN.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNetIBN.__factory[depth](pretrained=self.pretrained)
        
        print("Load the pretrained M2T...")
        import pickle as pkl
        ckpt = '/dev/shm/unsupervised_pretrained_m2t_ibn.pkl'
        print("loading ckpt... ",ckpt)
        para = pkl.load(open(ckpt, 'rb'), encoding='utf-8')['model']
        resnet.load_state_dict(para,strict = False)
        
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        gap = GeneralizedMeanPoolingP() #nn.AdaptiveAvgPool2d(1)
        print("The init norm is ",gap)
        waveblock = Waveblock()
        
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, resnet.relu,
            resnet.layer1,
            resnet.layer2, waveblock,
            resnet.layer3, waveblock,
            resnet.layer4, gap
        ).cuda()
        
        if not self.cut_at_pooling:
            self.num_features = 2048
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = build_metric('cos', 2048, self.num_classes, s=64, m=0.35).cuda()
                
            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = 2048
                feat_bn = nn.BatchNorm1d(self.num_features)
                
            feat_bn.bias.requires_grad_(False)

            
        init.constant_(feat_bn.weight, 1)
        init.constant_(feat_bn.bias, 0)
        
        self.projector_feat_bn = nn.Sequential(
            feat_bn
        ).cuda()
        
        
    def forward(self, x, y=None):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        bn_x =self.projector_feat_bn(x)
        prob = self.classifier(bn_x, y)
        
        # <-Split FC
        return x, prob


def resnet_ibn50a(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNetIBN('101a', **kwargs)

from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import random
from collections import OrderedDict

from .metric import build_metric

from timm.models import create_model
import models.vit_models

__all__ = ['VisionTransformer', 'vit_tiny', 'vit_small', 'vit_base','vit_tiny_pattern', 'vit_small_pattern', 'vit_base_pattern', 'vit_large_pattern']


class VisionTransformer(nn.Module):

    def __init__(self, weight, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(VisionTransformer, self).__init__()
        self.pretrained = True
        self.weight = weight
        self.cut_at_pooling = cut_at_pooling
        self.num_classes = num_classes
        
        if self.weight == 'tiny':
            vit = create_model(
                'deit_tiny_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        elif self.weight == 'small':
            vit = create_model(
                'deit_small_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        elif self.weight == 'base':
            vit = create_model(
                'deit_base_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
        
        elif self.weight == 'tiny_pattern':
            vit = create_model(
                'deit_tiny_pattern_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
        
        elif self.weight == 'small_pattern':
            vit = create_model(
                'deit_small_pattern_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        elif self.weight == 'base_pattern':
            vit = create_model(
                'deit_base_pattern_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        elif self.weight == 'large_pattern':
            vit = create_model(
                'deit_large_pattern_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        else:
            print("Not implement!!!")
            exit()
        
        vit.head = nn.Sequential()
        
        self.base = nn.Sequential(
            vit
        ).cuda()
        
        self.classifier = build_metric('cos', vit.embed_dim, self.num_classes, s=64, m=0.35).cuda()
        
    def forward(self, x, y = None):
        if 'pattern' not in self.weight:
            x = self.base(x)
            prob = self.classifier(x, y[0])
            return x, prob
        
        else:
            x, logits, pattern_matrix = self.base(x)
            #prob = self.classifier(x, y[0])
            #return x, prob

def vit_tiny(**kwargs):
    return VisionTransformer('tiny', **kwargs)

def vit_small(**kwargs):
    return VisionTransformer('small', **kwargs)

def vit_base(**kwargs):
    return VisionTransformer('base', **kwargs)

def vit_tiny_pattern(**kwargs):
    return VisionTransformer('tiny_pattern', **kwargs)

def vit_small_pattern(**kwargs):
    return VisionTransformer('small_pattern', **kwargs)

def vit_base_pattern(**kwargs):
    return VisionTransformer('base_pattern', **kwargs)

def vit_large_pattern(**kwargs):
    return VisionTransformer('large_pattern', **kwargs)
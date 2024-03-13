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
import models.vit_models_tune

__all__ = ['VisionTransformer','vit_tiny_pattern_tune', 'vit_small_pattern_tune', 'vit_base_pattern_tune', 'vit_large_pattern_tune']


class VisionTransformer(nn.Module):

    def __init__(self, weight, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(VisionTransformer, self).__init__()
        self.pretrained = True
        self.weight = weight
        self.cut_at_pooling = cut_at_pooling
        self.num_classes = num_classes
        
        if self.weight == 'tiny_pattern_tune':
            vit = create_model(
                'deit_tiny_pattern_patch16_224_tune',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
        
        elif self.weight == 'small_pattern_tune':
            vit = create_model(
                'deit_small_pattern_patch16_224_tune',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        elif self.weight == 'base_pattern_tune':
            vit = create_model(
                'deit_base_pattern_patch16_224_tune',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        elif self.weight == 'large_pattern_tune':
            vit = create_model(
                'deit_large_pattern_patch16_224_tune',
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
            x, logits = self.base(x)
            prob = self.classifier(x, y[0])
            return logits, prob

def vit_tiny_pattern_tune(**kwargs):
    return VisionTransformer('tiny_pattern_tune', **kwargs)

def vit_small_pattern_tune(**kwargs):
    return VisionTransformer('small_pattern_tune', **kwargs)

def vit_base_pattern_tune(**kwargs):
    return VisionTransformer('base_pattern_tune', **kwargs)

def vit_large_pattern_tune(**kwargs):
    return VisionTransformer('large_pattern_tune', **kwargs)
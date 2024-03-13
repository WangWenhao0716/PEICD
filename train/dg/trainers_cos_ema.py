from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import CrossEntropyLabelSmooth#, CosfacePairwiseLoss
from .utils.meters import AverageMeter
from .layer import MarginCosineProduct

class Trainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon=0).cuda()
        
    def train(self, epoch, data_loader, optimizer, ema, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_bce = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader.next()
            data_time.update(time.time() - end)

            s_inputs, targets_list = self._parse_data(source_inputs)
            logits, s_cls_out = self.model(s_inputs, targets_list)
            
            loss_ce, loss_bce, prec1 = self._forward(logits, s_cls_out, targets_list)
            loss = loss_ce + 0.5 * loss_bce

            losses_ce.update(loss_ce.item())
            losses_bce.update(loss_bce.item())
            
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'LR:{:.8f}\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_bce {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,optimizer.param_groups[0]["lr"],
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_bce.val, losses_bce.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fnames, pids, _, target_pattern = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        target_list = [targets, target_pattern]
        return inputs, target_list

    def _forward(self, logits, s_outputs, targets_list):
        logits = logits.cuda()
        s_outputs = s_outputs.cuda()
        for i in range(len(targets_list)):
            targets_list[i] = targets_list[i].cuda()
        
        loss_ce = self.criterion_ce(s_outputs, targets_list[0])
        prec, = accuracy(s_outputs.data, targets_list[0].data)
        prec = prec[0]
        
        pattern_target = targets_list[1].cuda() # B * num_pattern; 0, 1 label
        
        pos_weight = torch.ones([pattern_target.shape[1]])
        bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight).cuda()
        loss_bce = bce_criterion(logits, pattern_target) 

        return loss_ce, loss_bce, prec



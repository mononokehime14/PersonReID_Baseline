# encoding: utf-8


import torch.nn.functional as F
import torch.nn as nn
import torch

from .triplet_loss import TripletLoss
from .capsuleloss import CapsuleLoss


def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target) + triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'capsule':
        def loss_func(imgs, labels, results, y, y2):
            ''' 
            results, y and y2 are outputs of the 3 branches
            in ReIDCaps network.
            '''
            alpha = 0.5
            criterion = CapsuleLoss().to(cfg.DEVICE)
            criterion2 = nn.CrossEntropyLoss().to(cfg.DEVICE)
            criterion3 = nn.CrossEntropyLoss().to(cfg.DEVICE)
            one_hot_labels = torch.eye(632).to(cfg.DEVICE).index_select(dim=0, index=labels)
            loss = criterion(imgs, one_hot_labels, results) + alpha*criterion2(y, labels) + alpha*criterion3(y2, labels)
            return loss
        
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func

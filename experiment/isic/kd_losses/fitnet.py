from __future__ import print_function
import torch.nn as nn
import pdb


class FitNet(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self, weight, _layers, module_list, lambd=0.01, device='cuda:0'):
        super(FitNet, self).__init__()
        self.lamdb = lambd
        self.crit = nn.MSELoss()
        self.layers = _layers
        self.module_list = module_list
        self.loss_base = nn.CrossEntropyLoss(weight=weight).to(device)

    def forward(self, out_s, out_t, target):
        feat_s = out_s[self.layers['_layer_s']]
        feat_s_conv = self.module_list[1](feat_s) 
        logits_s = out_s['logits']
        feat_t = out_t[self.layers['_layer_t']]

        loss = self.loss_base(logits_s, target)

        loss = self.crit(feat_s_conv, feat_t) + self.lamdb * loss
        return loss
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class BLKD(nn.Module):
    def __init__(self, weight, lambd=0.5, T=5, device='cuda:0'):
        super(BLKD, self).__init__()
        self.lamdb = lambd
        self.T = T
        self.loss_base = nn.CrossEntropyLoss(weight=weight).to(device)
        self.loss_kd = nn.KLDivLoss(size_average=False).to(device)

    def forward(self, out_s, out_t, target):
        logits_s = out_s['logits']
        logits_t = out_t['logits']
        loss = self.loss_base(logits_s, target)
        batch_size = target.shape[0]

        s_max = F.log_softmax(logits_s / self.T, dim=1)
        t_max = F.softmax(logits_t / self.T, dim=1)
        loss_kd = self.loss_kd(s_max, t_max) / batch_size
        loss = (1 - self.lamdb) * loss + self.lamdb * self.T * self.T * loss_kd     
        return loss


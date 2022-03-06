'''
From https://github.com/lenscloth/RKD/blob/master/metric/loss.py
https://raw.githubusercontent.com/AberHu/Knowledge-Distillation-Zoo/master/kd_losses/rkd.py
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class DRKD(nn.Module):
    '''
    Relational Knowledge Distillation
    https://arxiv.org/pdf/1904.05068.pdf
    '''

    def __init__(self, weight, w_dist=25.0, w_angle=50.0, lambd=3.0, device='cuda:0'):
        super(DRKD, self).__init__()

        self.lamdb = lambd
        self.w_dist = w_dist
        self.w_angle = w_angle
        self.loss_base = nn.CrossEntropyLoss(weight=weight).to(device)

    def forward(self, out_s, out_t, target):
        feat_s = out_s['avg_pool']
        feat_t = out_t['avg_pool'] 
        logits_s = out_s['logits']
        loss = self.loss_base(logits_s, target)
        loss = self.w_dist * self.rkd_dist(feat_s, feat_t) + \
            self.w_angle * self.rkd_angle(feat_s, feat_t) + self.lamdb * loss
        return loss

    def rkd_dist(self, feat_s, feat_t):
        with torch.no_grad():
            feat_t_dist = self.pdist(feat_t, squared=False)
            mean_feat_t_dist = feat_t_dist[feat_t_dist > 0].mean()
            feat_t_dist = feat_t_dist / mean_feat_t_dist

        feat_s_dist = self.pdist(feat_s, squared=False)
        mean_feat_s_dist = feat_s_dist[feat_s_dist > 0].mean()
        feat_s_dist = feat_s_dist / mean_feat_s_dist

        loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

        return loss

    def rkd_angle(self, feat_s, feat_t):
        # N x C --> N x N x C
        with torch.no_grad():
            feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
            norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
            feat_t_angle = torch.bmm(
                norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)

        feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
        norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
        feat_s_angle = torch.bmm(
            norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

        return loss

    def pdist(self, feat, squared=False, eps=1e-12):
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) +
                     feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

        if not squared:
            feat_dist = feat_dist.sqrt()

        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0

        return feat_dist

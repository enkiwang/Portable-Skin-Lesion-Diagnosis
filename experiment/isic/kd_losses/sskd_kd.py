from __future__ import print_function
import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F


def gram_matrix(features, normalize=True):

    N, C, H, W = features.shape
    feats = features.view(N, C, H * W)
    feats_t = feats.transpose(1, 2)
    G = feats.bmm(feats_t)

    if normalize:
        return G.div(C * H * W)
    else:
        return G


class D_KD(nn.Module):
    def __init__(self, weight, _layers, module_list, lambd=0.5, T=5, 
                lambd_rkd=1.0, w_dist=25.0, w_angle=50.0, 
                lambd_crkd=1e4, device='cuda:0'):
        super(D_KD, self).__init__()
        self.lambd = lambd
        self.lambd_rkd = lambd_rkd
        self.lambd_crkd = lambd_crkd
        self.T = T
        self.w_dist = w_dist
        self.w_angle = w_angle

        self.crit = nn.MSELoss()
        self.layers = _layers
        self.module_list = module_list
        self.loss_base = nn.CrossEntropyLoss(weight=weight).to(device)
        self.loss_blkd = nn.KLDivLoss(size_average=False).to(device)

    def forward(self, out_s, out_t, target):
        feat_s = out_s[self.layers['_layer_s']]
        feat_t = out_t[self.layers['_layer_t']]
        feat_s_conv = self.module_list[1](feat_s) 

        pool_s = out_s['avg_pool']
        pool_t = out_t['avg_pool']

        logits_s = out_s['logits']
        logits_t = out_t['logits']

        batch_size = target.shape[0]
        s_max = F.log_softmax(logits_s / self.T, dim=1)
        t_max = F.softmax(logits_t / self.T, dim=1)

        loss_cls = self.loss_base(logits_s, target) 
        loss_blkd = self.loss_blkd(s_max, t_max) / batch_size   
        loss_rkd = self.w_dist * self.rkd_dist(pool_s, pool_t) + \
                    self.w_angle * self.rkd_angle(pool_s, pool_t)  
        loss_crkd = self.cr_dist(feat_s_conv, feat_t)  
 
        loss = (1 - self.lambd) * loss_cls + self.lambd * self.T * self.T * loss_blkd + \
                    self.lambd_rkd * loss_rkd + self.lambd_crkd * loss_crkd 
        return loss

    def cr_dist(self, feat_s, feat_t): 
        gram_s = gram_matrix(feat_s)
        gram_t = gram_matrix(feat_t)
        cr_dist_tmp = F.mse_loss(gram_s, gram_t)
        return cr_dist_tmp.sqrt()


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


class DR(D_KD):
    def __init__(self, weight, _layers, module_list, lambd=0.5, T=5, 
                lambd_rkd=1.0, w_dist=25.0, w_angle=50.0, 
                lambd_crkd=1e4, device='cuda:0'):
        super(DR, self).__init__()
        self.lambd = lambd
        self.lambd_rkd = lambd_rkd
        self.lambd_crkd = lambd_crkd
        self.T = T
        self.w_dist = w_dist
        self.w_angle = w_angle

        self.layers = _layers
        self.module_list = module_list

    def forward(self, out_s, out_t, target=None):
        feat_s = out_s[self.layers['_layer_s']]
        feat_t = out_t[self.layers['_layer_t']]
        feat_s_conv = self.module_list[2](feat_s) 

        pool_s = out_s['avg_pool']
        pool_t = out_t['avg_pool']

        loss_rkd = self.w_dist * self.rkd_dist(pool_s, pool_t) + \
                    self.w_angle * self.rkd_angle(pool_s, pool_t)  
        loss_crkd = self.cr_dist(feat_s_conv, feat_t)  
 
        loss = self.lambd_rkd * loss_rkd + self.lambd_crkd * loss_crkd 
        return loss


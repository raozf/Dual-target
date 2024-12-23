import copy
import numpy as np
import open3d as o3d
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)

from utils import batch_transform
from models import Coarse2, Fine, gather_points, weighted_icp


class Dual_target(nn.Module):
    def __init__(self, args):
        super(Dual_target, self).__init__()
        self.N1 = args.test_N1
        self.use_ppf = args.use_ppf
        self.cg= Coarse2(in_dim=3, gn=True)
        self.tfmr = Fine(args)

    def forward(self, src,src1,src2,tgt, num_iter=1, train=False):
        '''
        :param src: (B, N, 3) or (B, N, 6) [normal for the last 3]
        :param tgt: (B, M, 3) or (B, M, 6) [normal for the last 3]
        :param num_iter: int, default 1.
        :param train: bool, default False
        :return: dict={'T': (B, 3, 4),
                        '': }
        '''
        B, N, C = src.size()
        normal_src, normal_tgt = None, None
        if self.use_ppf and C == 6:
            normal_src = src[..., 3:]
            normal_tgt = tgt[..., 3:]
        src = src[..., :3]
        src1 = src1[..., :3]
        src2 = src2[..., :3]
        src_raw = copy.deepcopy(src)
        tgt = tgt[..., :3]

        results = {}
        pred_Ts, pred_src, pred_src_R,pred_src2, pred_src_R2 = [], [], [],[],[]

        # CG module
        T0,T1,T2, x_ol, y_ol = self.cg(src,src1,src2,tgt)
        R, t = T0[:, :3, :3], T0[:, :3, 3]
        R1, t1 = T1[:, :3, :3], T1[:, :3, 3]
        R2, t2 = T2[:, :3, :3], T2[:, :3, 3]
        
        src_t = batch_transform(src_raw, R, t)
        src_t_R = batch_transform(src_raw, R)
        
        src_t2 = batch_transform(src1, R1, t1)
        src_t_R2 = batch_transform(src1, R1)
        pred_src2.append(src_t2)
        pred_src_R2.append(src_t_R2)
        src_t3 = batch_transform(src2, R2, t2)
        src_t_R3 = batch_transform(src2, R2)
        pred_src2.append(src_t3)
        pred_src_R2.append(src_t_R3)
        
        normal_src_t = None
        if normal_src is not None:
            normal_src_t = batch_transform(normal_src, R).detach()
            normal_src_R = batch_transform(normal_src, R).detach()
        pred_Ts.append(T0)
        pred_src.append(src_t)
        pred_src_R.append(src_t_R)
        x_ol_score = torch.softmax(x_ol, dim=1)[:, 1, :].detach()  # (B, N)
        y_ol_score = torch.softmax(y_ol, dim=1)[:, 1, :].detach()  # (B, N)
        
        for i in range(num_iter):
            src_t = src_t.detach()
            src_t_R = src_t_R.detach()
            src, tgt_corr, icp_weights, similarity_max_inds = \
                self.tfmr(src=src_t,
                          tgt=tgt,
                          x_ol_score=x_ol_score,
                          y_ol_score=y_ol_score,
                          train=train,
                          iter=i,
                          normal_src=normal_src_t,
                          normal_tgt=normal_tgt)

            R_cur, t_cur, _ = weighted_icp(src=src,
                                           tgt=tgt_corr,
                                           weights=icp_weights)
            R, t = R_cur @ R, R_cur @ t[:, :, None] + t_cur[:, :, None]
            
            T = torch.cat([R, t], dim=-1)
            pred_Ts.append(T)
            src_t = batch_transform(src_raw, R, torch.squeeze(t, -1))
            src_t_R = batch_transform(src_raw, R)
            pred_src.append(src_t)
            pred_src_R.append(src_t_R)
            normal_src_t = batch_transform(normal_src, R).detach()
            normal_src_t_R = batch_transform(normal_src, R).detach()
            t = torch.squeeze(t, dim=-1)

        ## for overlapping points in src
        _, x_ol_inds = torch.sort(x_ol_score, dim=-1, descending=True)
        x_ol_inds = x_ol_inds[:, :self.N1]
        src_ol1 = gather_points(src_raw, x_ol_inds)
        src_ol2 = gather_points(src_ol1, similarity_max_inds)

        results['pred_Ts'] = pred_Ts
        results['pred_src'] = pred_src
        results['pred_src_R'] = pred_src_R
        results['pred_src2'] = pred_src2
        results['pred_src_R2'] = pred_src_R2
        results['x_ol'] = x_ol
        results['y_ol'] = y_ol
        results['src_ol1'] = src_ol1
        results['src_ol2'] = src_ol2

        return results
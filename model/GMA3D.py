from dis import dis
from math import dist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gma3D(nn.Module):
    """
    context_features: (B,N,64)
    motion_features: (B,N,64)
    Wq:(64,64/4)
    Wk:(64,64/4)
    Wv:(64,64)
    question:ues position encoder use residual attention

    """

    def __init__(self, context_features_dim=64, motion_features_dim=64, gma_dim=64):
        # set alpha to weight global motion aggregation
        super(Gma3D, self).__init__()
        self.sa = self_attention(channels=context_features_dim)

        self.alpha_gma = nn.Parameter(torch.zeros(1))
        self.v_conv = nn.Conv1d(motion_features_dim, motion_features_dim, 1)
        self.trans_conv = nn.Conv1d(motion_features_dim, motion_features_dim, 1)
        self.group_norm = nn.GroupNorm(8, motion_features_dim)
        self.relu = nn.PReLU()

    def forward(self, context_features, motion_features, xyz1=None):
        # b, c, n
        attention_matrix = self.sa(context_features, xyz1)
        x_v = self.v_conv(motion_features)

        gma = torch.bmm(x_v, attention_matrix)

        # residual attention
        gma_residual = self.trans_conv(motion_features - gma)
        gma_res = self.relu(self.group_norm(gma_residual))
        # gma_no_res = self.relu(self.group_norm(self.trans_conv(gma)))
        
        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r))) x = x + x_r
        gma3D = self.alpha_gma * gma_res + motion_features
        
        return gma3D


class self_attention(nn.Module):
    def __init__(self, channels):
        super(self_attention, self).__init__()
        # add position 
        # self.pos_encoder = nn.Conv1d(3, context_features_dim, 1)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        #  add pos encoder
        self.am_conv = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.GroupNorm(8, 16),
            nn.PReLU(),
            nn.Conv2d(16, 1, 1),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, context_features, xyz1, pos_coef=0.1):
        # b, n, c
        # x = x + xyz
        x = context_features
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)

        # b, n, n
        corr_m = torch.bmm(x_q, x_k)

        attention = self.softmax(corr_m)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # add pos distance attention
        #pos_alpha = 1
        '---------------------------pos attention------------------------'
        b, np, _ = xyz1.size()

        p = xyz1.view(b, 1, np, 3).expand(b, np, np, 3)
        p_t = p.permute(0, 2, 1, 3)
        dist = p - p_t
        dist = torch.sum(dist ** 2, dim=-1)
        pos_am = dist / (1e-9 + dist.sum(dim=1, keepdim=True))
        dis_mask = pos_am <= pos_coef
        local_attention = attention * dis_mask
        local_attention = self.softmax(local_attention)
        local_attention = local_attention / (1e-9 + local_attention.sum(dim=1, keepdim=True))
        '---------------------------pos attention------------------------'
        
        # add pos_attention
        attention = attention + local_attention
        
        return attention

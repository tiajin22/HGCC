import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .sc_encoder import Sc_encoder
from .feat_encoder import Feat_encoder
from .contrast import Contrast
import torch as th
import numpy as np


class HGCC(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 sample_feat_rate, nei_num, tau, lam):
        super(HGCC, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, hidden_dim, attn_drop)
        self.sc = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)
        self.feat_simi = Feat_encoder(hidden_dim, sample_feat_rate, attn_drop)
        self.contrast = Contrast(hidden_dim, tau, lam)


    def forward(self, feats, pos, mps, nei_index, feat_nei, feat_adj, num_clusters):  # feat=[p a s]
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]).cpu()).cuda()))
            # h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        # print(h_all)
        z_mp = self.mp(h_all[0], mps)
        z_sc = self.sc(h_all, nei_index)

        z_feat = self.feat_simi(h_all[0], feat_nei)  
        # z_feat = self.feat_simi(h_all[0], feat_adj) 

        # loss = self.contrast(z_mp, z_sc, pos, num_clusters)
        # loss = self.contrast(z_mp, z_sc,h_all[0], pos, num_clusters) 
        loss = self.contrast(z_mp, z_sc, z_feat, pos, num_clusters)
        return loss

    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, mps)
        return z_mp.detach()

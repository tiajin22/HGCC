import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj.cpu(), seq_fts.cpu()).cuda()
        # out = torch.spmm(adj.cpu(), seq_fts.cpu())
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        print("sc ", beta.data.cpu().numpy())  # type-level attention
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):
        nei_emb = F.embedding(nei, h)
        h_refer = torch.unsqueeze(h_refer, 1)
        h_refer = h_refer.expand_as(nei_emb)
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)
        nei_emb = (att*nei_emb).sum(dim=1)
        return nei_emb



class Feat_encoder(nn.Module):
    def __init__(self, hidden_dim, sample_feat_rate, attn_drop):
        super(Feat_encoder, self).__init__()
        self.intra = nn.ModuleList([intra_att(hidden_dim, attn_drop)])
        self.sample_feat_rate = sample_feat_rate

    def forward(self, feat, feat_nei):
        sele_nei = []
        sample_num = self.sample_feat_rate
        count = 0
        for per_node_nei in feat_nei:
            if len(per_node_nei) >= sample_num:
                select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                           replace=False))[np.newaxis]
            else:
                try:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                            replace=True))[np.newaxis]  #可反复选同一个元素
                except:
                    a=0
            sele_nei.append(select_one)
        count += 1
        sele_nei = torch.cat(sele_nei, dim=0).cuda() #拼接
        z_feat = F.elu(self.intra[0](sele_nei, feat, feat))

        # z_feat = F.elu(self.intra[0](feat_nei, feat, feat))
        return z_feat



# class Feat_encoder(nn.Module):
#     def __init__(self, hidden_dim, sample_feat_rate, attn_drop):
#         super(Feat_encoder, self).__init__()
#         self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim)])
#
#     def forward(self, feat, feat_adj):
#         z_feat = self.node_level[0](feat, feat_adj)
#         return z_feat
import torch
import torch.nn as nn
from .cluster import *
from torch.nn.functional import cosine_similarity

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj_sc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414) #采用正态分布填充

    def sim(self, z1, z2, tau):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t()).cuda()
        dot_denominator = torch.mm(z1_norm, z2_norm.t()).cuda()
        sim_matrix = torch.exp(dot_numerator / dot_denominator / tau).cuda()
        return sim_matrix #exp(sim(z_sc,z_mp)/tau)

    def POCL(self, z_anchor, z, cluster_result, num_clusters):
        loss = torch.tensor(0.0, requires_grad=True).cuda()
        # node_prototypes tensor(4019,64)
        im2cluster = cluster_result['im2cluster']
        im2cluster = torch.stack(im2cluster)
        k_times = len(num_clusters)
        N = z_anchor.size(0)
        weight = torch.zeros([N, N])
        for i in range(k_times):
            # node_idx = torch.range(0,N-1).long().cuda()
            node_idx = torch.range(0, N - 1).long()
            cluster_idx = im2cluster[i]
            cluster_idx = cluster_idx.cpu()
            idx = torch.stack((node_idx, cluster_idx), 0)
            # data = torch.ones(N).cuda()
            data = torch.ones(N)
            coo_i = torch.sparse_coo_tensor(idx, data, [N, num_clusters[i]])
            weight = weight + torch.mm(coo_i, coo_i.to_dense().t())
        #
        for i, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], \
                                                                  cluster_result['centroids'],
                                                                  cluster_result['density'])):
        #
            weight_clus = weight
            diff_indices = torch.nonzero(im2cluster[:,None] != im2cluster)
            weight_clus[diff_indices[:,0], diff_indices[:,1]] = 0
            weight_clus = weight_clus.cuda()
            c = torch.matmul(weight_clus, z) / torch.sum(weight_clus, dim=1)[:, None]
            # node_prototypes = prototypes[im2cluster]
            node_prototypes = 1 * node_prototypes + 0.1 * c  # tensor(4019,64)

            # node_prototypes = prototypes[im2cluster]  # tensor(4019,64)
            phi = density[im2cluster]
            pos_prototypes = torch.exp(torch.mul(z_anchor, node_prototypes).cuda().sum(axis=1) / phi) 
            neg_prototypes = torch.exp(torch.mm(z_anchor, prototypes.t()).cuda() / density).mean(axis=1) * z_anchor.size(0) 
            loss = loss + ((-1) * (torch.log(pos_prototypes / neg_prototypes).cuda())).mean()
        loss = loss / len(cluster_result['im2cluster'])
        return loss

    def Proto(self, z_anchor, z, num_clusters):
        cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
        for num_cluster in num_clusters:
            cluster_result['im2cluster'].append(torch.zeros(z_anchor.size(0), dtype=torch.long).cuda())
            cluster_result['centroids'].append(torch.zeros(num_cluster, z_anchor.size(1)).cuda())
            cluster_result['density'].append(torch.zeros(num_cluster).cuda())
        cluster_result = run_kmeans(z, num_clusters, 0.6)
        loss_proto = self.POCL(z_anchor, z, cluster_result, num_clusters)
        return loss_proto

    def forward(self, z_mp, z_sc, feat, pos, num_clusters):
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc, self.tau)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()
        # print(lori_mp)

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        # print(lori_sc)

        # POCL
        proto_loss_sc2mp = self.Proto(z_sc, z_mp, num_clusters)
        # print(proto_loss_sc2mp)

        # intra
        mp_intra  = self.proj_mp(z_mp)
        # sc_intra = self.proj_sc(z_sc)
        # intra_sc = self.sim(sc_intra, sc_intra, 0.1)
        intra_mp = self.sim(mp_intra, mp_intra, 0.05)

        # matrix_sc2sc = intra_sc / (torch.sum(intra_sc, dim=1).view(-1, 1) + 1e-8)
        matrix_mp2mp = intra_mp / (torch.sum(intra_mp, dim=1).view(-1, 1) + 1e-8)

        # loss_intrasc = -torch.log(matrix_sc2sc.mul(pos.to_dense()).sum(dim=-1)).mean()
        loss_intramp = -torch.log(matrix_mp2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        # print(loss_intramp)

        # feat
        proj_mp_feat = self.proj(z_mp)
        proj_feat = self.proj(feat)

        matrix_mp2feat = self.sim(proj_mp_feat, proj_feat, self.tau)
        matrix_mp2feat = matrix_mp2feat/ (torch.sum(matrix_mp2feat, dim=1).view(-1, 1) + 1e-8)
        lori_mpfeat = -torch.log(matrix_mp2feat.mul(pos.to_dense()).sum(dim=-1)).mean()
        # print(lori_mpfeat)

        matrix_feat2mp = self.sim(proj_feat, proj_mp_feat, self.tau)
        matrix_feat2mp = matrix_feat2mp / (torch.sum(matrix_feat2mp, dim=1).view(-1, 1) + 1e-8)
        lori_featmp = -torch.log(matrix_feat2mp.mul(pos.to_dense()).sum(dim=-1)).mean()

        # return self.lam * lori_mp + (1 - self.lam) * lori_sc + 0.5 * proto_loss_sc2mp + 1 * loss_intramp + 0.5 * lori_featmp + 0.5 * lori_mpfeat#  #+ 0.5 * proto_loss_sc2mp
        return self.lam * lori_mp + (1 - self.lam) * lori_sc  + 0.3 * loss_intramp + 1 * proto_loss_sc2mp + 0.01 * lori_mpfeat + 0.01 * lori_featmp #+ 1 * proto_loss_sc2mp

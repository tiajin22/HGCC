import torch as th
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

# path = "../data/acm/"
# feat = sp.load_npz(path + "p_feat.npz")
# feat = th.FloatTensor(preprocess_features(feat))
# feat = feat / th.norm(feat, dim=-1, keepdim=True) 
# feat = feat.cpu()
# similarity = th.mm(feat, feat.T) 
# mean = th.mean(similarity)
# # mean = 0.15
# feat = [th.nonzero(row >= mean).reshape(-1) for row in similarity]
#
# mask = np.zeros([len(feat),len(feat)], dtype=bool)
# for i in range(mask.shape[0]):
#     mask[i][feat[i]] = True
# feat_adj = sp.coo_matrix(mask)
# sp.save_npz("acm/feat_adj.npz", feat_adj)
#
# feat_nei = [np.array(i) for i in feat]
# feat_nei = np.array(feat_nei, dtype=object)
# np.save("acm/feat_nei.npy", feat_nei)


# path = "../data/freebase/"
# # feat_m = sp.eye(3492)
# feat = sp.load_npz(path + "feat_m.npz")
# feat = th.FloatTensor(preprocess_features(feat))
# feat = feat / th.norm(feat, dim=-1, keepdim=True)  
# feat = feat.cpu()
# similarity = th.mm(feat, feat.T)  
# mean = th.mean(similarity)
# # mean = 0.24
# feat = [th.nonzero(row >= mean).reshape(-1) for row in similarity]
#
# mask = np.zeros([len(feat),len(feat)], dtype=bool)
# for i in range(mask.shape[0]):
#     mask[i][feat[i]] = True
# feat_adj = sp.coo_matrix(mask)
# sp.save_npz("freebase/feat_adj.npz", feat_adj)
#
# feat_nei = [np.array(i) for i in feat]
# feat_nei = np.array(feat_nei, dtype=object)
# np.save("freebase/feat_nei.npy", feat_nei)

# path = "../data/aminer/"
# feat = sp.load_npz(path + "p_feat.npz")
# print(feat)
# feat = th.FloatTensor(preprocess_features(feat))
# print(feat)
# feat = feat / th.norm(feat, dim=-1, keepdim=True) 
# feat = feat.cpu()
# similarity = th.mm(feat, feat.T)  
# #mean = th.mean(similarity)
# mean = 0.3
# feat = [th.nonzero(row >= mean).reshape(-1) for row in similarity]
#
# mask = np.zeros([len(feat),len(feat)], dtype=bool)
# for i in range(mask.shape[0]):
#     mask[i][feat[i]] = True
# feat_adj = sp.coo_matrix(mask)
# sp.save_npz("aminer/feat_adj.npz", feat_adj)
#
# feat_nei = [np.array(i) for i in feat]
# feat_nei = np.array(feat_nei, dtype=object)
# np.save("aminer/feat_nei.npy", feat_nei)

# path = "../data/dblp/"
# feat = sp.load_npz(path + "a_feat.npz")
# print(feat)
# feat = th.FloatTensor(preprocess_features(feat))
# print(feat)
# feat = feat / th.norm(feat, dim=-1, keepdim=True) 
# feat = feat.cpu()
# similarity = th.mm(feat, feat.T)  
# #mean = th.mean(similarity)
# mean = 0.3
# feat = [th.nonzero(row >= mean).reshape(-1) for row in similarity]
#
# mask = np.zeros([len(feat),len(feat)], dtype=bool)
# for i in range(mask.shape[0]):
#     mask[i][feat[i]] = True
# feat_adj = sp.coo_matrix(mask)
# sp.save_npz("dblp/feat_adj.npz", feat_adj)
#
# feat_nei = [np.array(i) for i in feat]
# feat_nei = np.array(feat_nei, dtype=object)
# np.save("dblp/feat_nei.npy", feat_nei)

path = "../data/imdb/"
feat = sp.load_npz(path + "features_0.npz")
feat = th.FloatTensor(preprocess_features(feat))
feat = feat / th.norm(feat, dim=-1, keepdim=True)  
feat = feat.cpu()
similarity = th.mm(feat, feat.T)  
# mean = th.mean(similarity)
mean = 0.3
feat = [th.nonzero(row > mean).reshape(-1) for row in similarity]
mask = np.zeros([len(feat),len(feat)], dtype=bool)
for i in range(mask.shape[0]):
    mask[i][i] = True
    mask[i][feat[i]] = True
feat_adj = sp.coo_matrix(mask)
sp.save_npz("imdb/feat_adj.npz", feat_adj)

feat_nei = [np.array(i) for i in feat]
feat_nei = np.array(feat_nei, dtype=object)
np.save("imdb/feat_nei.npy", feat_nei)
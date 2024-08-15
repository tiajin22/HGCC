import numpy as np
import scipy.sparse as sp
import torch
import torch as th

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)) 
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def row_normalize_matrix(adj):
    """Row-normalize feature matrix"""
    rowsum = np.array(adj.sum(1), dtype=np.float32)
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt = np.reshape(d_inv_sqrt, [-1,1])
    adj = np.multiply(adj, d_inv_sqrt)
    adj = sp.coo_matrix(adj)
    return adj.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32) 
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def personal_pagerank(adj_matrix, k, alpha=0.85):
    n = adj_matrix.shape[0]
    p = np.ones(n) / n
    for i in range(20):
        new_p = (1 - alpha) * np.dot(adj_matrix, p) + alpha * p
        if np.allclose(new_p, p):
            break
        p = new_p
    top_k_matrix = th.from_numpy(np.multiply(adj_matrix, p))
    value, indices = th.topk(top_k_matrix, k)
    indices = np.array(indices)
    indices_addself = []
    for i in range(indices.shape[0]):
        indices_addself.append(np.append(indices[i],i))
    indices = th.from_numpy(np.array(indices_addself))
    mask = np.zeros(adj_matrix.shape, dtype=bool)
    for i in range(adj_matrix.shape[0]):
        mask[i][indices[i]] = True
    adj_matrix = adj_matrix & mask
    return adj_matrix

path = "acm/"
p_feat = sp.load_npz(path + "p_feat.npz")
p_feat = th.FloatTensor(preprocess_features(p_feat))

pap = sp.load_npz(path + "pap.npz")
# pap = sp.load_npz(path + "psp.npz")

choose_index = 8

pap_array = pap.toarray()
# pap_origin = []
# for i in range(4019):
#     if(pap_array[choose_index][i]!=0):
#         pap_origin.append(i)
# print("pap_origin: ", pap_origin)
#
# # pap_sp = sparse_mx_to_torch_sparse_tensor(pap)
# # print(pap_sp[choose_index])
# pap_sp1 = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
# print(pap_sp1[choose_index])
#
# pap_origin = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(pap))

# alpha  = 0.85
# k=10
# n = pap_array.shape[0]
# p = np.ones(n) / n
# for i in range(20):
#     new_p = (1 - alpha) * np.dot(pap_array, p) + alpha * p
#     if np.allclose(new_p, p):
#         break
#     p = new_p
# top_k_matrix = th.from_numpy(np.multiply(pap_array, p))
# print(top_k_matrix[0])
# value, indices = th.topk(top_k_matrix, k)
# print(indices[0])
# print(value[0])
# mask = np.zeros(pap_array.shape, dtype=bool)
# for i in range(pap_array.shape[0]):
#     mask[i][indices[i]] = True
# pap_matrix = pap_array & mask
pap_matrix = personal_pagerank(pap_array,10)
print(pap_matrix[0])
pap_matrix = sparse_mx_to_torch_sparse_tensor(row_normalize_matrix(pap_matrix))
# print(pap_origin[choose_index])
print(pap_matrix[choose_index])
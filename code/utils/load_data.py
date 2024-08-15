import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def row_normalize_matrix(adj):
    """Row-normalize feature matrix"""
    rowsum = np.array(adj.sum(1), dtype=np.float32)
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt = np.reshape(d_inv_sqrt, [-1,1])
    adj = np.multiply(adj, d_inv_sqrt)
    adj = sp.coo_matrix(adj)
    return adj.tocoo()
def matapath_denoising(feat, mp, deno_method, trunca_num):
    mp_array = mp.toarray()
    if(deno_method == "cosine"):
        mp_sp = sparse_mx_to_torch_sparse_tensor(mp)
        feat = feat / th.norm(feat, dim=-1, keepdim=True)  
        similarity = th.mm(feat, feat.T)  
        similarity_p = th.mul(similarity, mp_sp.to_dense())
        value, indices = th.topk(similarity_p, trunca_num)
    elif(deno_method == "PPR"):
        alpha = 0.85
        n = mp_array.shape[0]
        p = np.ones(n) / n
        for i in range(20):
            new_p = (1 - alpha) * np.dot(mp_array, p) + alpha * p
            if np.allclose(new_p, p):
                break
            p = new_p
        similarity_p = th.from_numpy(np.multiply(mp_array, p))
        value, indices = th.topk(similarity_p, trunca_num)
        indices = np.array(indices)
        indices_addself = []
        for i in range(indices.shape[0]):
            indices_addself.append(np.append(indices[i], i))
        indices = th.from_numpy(np.array(indices_addself))
    mask = np.zeros(mp_array.shape, dtype=bool)
    for i in range(mp_array.shape[0]):
        mask[i][indices[i]] = True
    mp = mp_array & mask
    return mp

def load_acm(ratio, type_num):
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_nei = np.load(path + "feat_nei.npy", allow_pickle=True)

    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pos = sp.load_npz(path + "pos.npz")
    feat_adj = sp.load_npz(path + "feat_adj.npz")

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]
    feat_nei = [th.LongTensor(i) for i in feat_nei]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))

    # pap = matapath_denoising(feat_p, pap, "cosine", 30)
    psp = matapath_denoising(feat_p, psp, "cosine", 400)
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    # pap = sparse_mx_to_torch_sparse_tensor(row_normalize_matrix(pap))
    psp = sparse_mx_to_torch_sparse_tensor(row_normalize_matrix(psp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    feat_adj = sparse_mx_to_torch_sparse_tensor(feat_adj)

    train = [th.LongTensor(i) for i in train] 
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]


    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test, feat_nei, feat_adj

def load_aminer(ratio, type_num):
    # The order of node types: 0 p 1 a 2 r
    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    feat_nei = np.load(path + "feat_nei.npy", allow_pickle=True)

    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    feat_adj = sp.load_npz(path + "feat_adj.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]
    feat_nei = [th.LongTensor(i) for i in feat_nei]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_r = th.FloatTensor(preprocess_features(feat_r))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    feat_adj = sparse_mx_to_torch_sparse_tensor(feat_adj)
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test, feat_nei, feat_adj


def load_freebase(ratio, type_num):
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    feat_nei = np.load(path + "feat_nei.npy", allow_pickle=True)
# # nei_d
#     data = np.ones((3762,))
#     col = []
#     for i in range(nei_d.shape[0]):
#         col.extend(nei_d[i])
#     col = np.array(col)
#     row = []
#     origin = 0
#     for i in range(nei_d.shape[0]):
#         len = nei_d[i].shape[0]
#         for j in range(len):
#             row.append(origin)
#         origin += 1
#     row = np.array(row)
#     nei_d = sp.coo_matrix((data, (row, col)), shape=(3492, 3762))
# #nei_a
#     data = np.ones((65341,))
#     col = []
#     for i in range(nei_a.shape[0]):
#         col.extend(nei_a[i])
#     col = np.array(col)
#     row = []
#     origin = 0
#     for i in range(nei_a.shape[0]):
#         len = nei_a[i].shape[0]
#         for j in range(len):
#             row.append(origin)
#         origin += 1
#     row = np.array(row)
#     nei_a = sp.coo_matrix((data, (row, col)), shape=(3492, 65341))
# # nei_w
#     data = np.ones((6414,))
#     col = []
#     for i in range(nei_w.shape[0]):
#         col.extend(nei_w[i])
#     col = np.array(col)
#     row = []
#     origin = 0
#     for i in range(nei_w.shape[0]):
#         len = nei_w[i].shape[0]
#         for j in range(len):
#             row.append(origin)
#         origin += 1
#     row = np.array(row)
#     nei_w = sp.coo_matrix((data, (row, col)), shape=(3492, 6414))

    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    feat_adj = sp.load_npz(path + "feat_adj.npz")
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]
    feat_nei = [th.LongTensor(i) for i in feat_nei]
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    feat_adj = sparse_mx_to_torch_sparse_tensor(feat_adj)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test, feat_nei, feat_adj
def load_imdb(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = "../data/imdb/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    feat_nei = np.load(path + "feat_nei.npy", allow_pickle=True)
    feat_m = sp.load_npz(path + "features_0.npz")
    feat_d = sp.load_npz(path + "features_1.npz")
    feat_a = sp.load_npz(path + "features_2.npz")
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mam = mam.tocoo()
    mdm = mdm.tocoo()
    pos = sp.load_npz(path + "pos_10.npz")
    feat_adj = sp.load_npz(path + "feat_adj.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_d = [th.LongTensor(i) for i in nei_d]
    feat_nei = [th.LongTensor(i) for i in feat_nei]
    # feat_m = normalize(feat_m, norm='l1', axis=1)
    feat_m = th.FloatTensor(normalize(feat_m, norm='l1', axis=1).todense())
    feat_d = th.FloatTensor(normalize(feat_d, norm='l1', axis=1).todense())
    feat_a = th.FloatTensor(normalize(feat_a, norm='l1', axis=1).todense())

    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    feat_adj = sparse_mx_to_torch_sparse_tensor(feat_adj)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_d, nei_a], [feat_m, feat_d, feat_a], [mam, mdm], pos, label, train, val, test, feat_nei, feat_adj


def load_data(dataset, ratio, type_num):
    if dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num)
    elif dataset == "imdb":
        data = load_imdb(ratio, type_num)
    # elif dataset == "aminer-D_latest":
    #     data = load_aminer-D_latest(ratio, type_num)
    # elif dataset == "aminer-D_all":
    #     data = load_aminer-D_all(ratio, type_num)
    return data

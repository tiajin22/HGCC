import torch
import datetime
import pickle as pkl
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from utils import load_data, set_params, evaluate
import scipy.sparse as sp

def run_kmeans(x, y, k, starttime, dataset):
    estimator = KMeans(n_clusters=k)

    NMI_list = []
    ARI_list = []
    for _ in range(100):
        estimator.fit(x)
        y_pred = estimator.predict(x)
        n1 = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        a1 = adjusted_rand_score(y, y_pred)
        NMI_list.append(n1)
        ARI_list.append(a1)

    nmi = sum(NMI_list) / len(NMI_list)
    ari = sum(ARI_list) / len(ARI_list)
    # nmi = max(NMI_list)
    # ari = max(ARI_list)
    print('\t[Clustering] NMI: {:.2f}   ARI: {:.2f}'.format(np.round(nmi*100,2), np.round(ari*100,2)))
    # f = open("result_" + dataset + "_NMI&ARI.txt", "a")
    # f.write(str(starttime.strftime('%Y-%m-%d %H:%M'))+"\t NMI: " + str(np.round(nmi*100,4)) +\
    #      "\t ARI: " + str(np.round(ari*100,4)) + "\n")
    # f.close()

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    dataset = "acm"
    args = set_params()
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test, feat_nei, feat_adj = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    print(dataset)
    print(" nb_classes")
    print(nb_classes)
    label = label.cuda()
    # print(label.shape)  #[4019,3] acm
    # print(len(idx_train))  #3
    # print(len(idx_val))  #3
    # print(len(idx_test))  #3

    file = open("./embeds/"+dataset+"/"+"new_en"+".pkl","rb")
    embeds = torch.from_numpy(pkl.load(file)).cuda()
    # embeds = sp.csr_matrix(pkl.load(file))
    # sp.save_npz("../data/freebase/feat_m.npz", embeds)
    run_kmeans(embeds.cpu(), torch.argmax(label.cpu(), dim=-1), nb_classes, starttime, dataset)


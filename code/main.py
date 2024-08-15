import numpy
import torch
from utils import load_data, set_params, evaluate
from module import HGCC
import warnings
import datetime
import pickle as pkl
import os
import random
from finalClsuter import run_kmeans
import time as Ti

warnings.filterwarnings('ignore')
args = set_params()
# device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

torch.cuda.empty_cache()
own_str = args.dataset

seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train():
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test, feat_nei, feat_adj = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))
    model = HGCC(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                    P, args.sample_rate, args.sample_feat_rate, args.nei_num, args.tau, args.lam).to(device)
    print(model)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef) 

    if torch.cuda.is_available():
        print('Using CUDA ' + str(args.gpu))
        model.cuda()
        feats = [feat.cuda() for feat in feats]
        mps = [mp.cuda() for mp in mps]
        pos = pos.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0
    starttime = datetime.datetime.now()
    num_clusters = args.num_cluster
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        print("...............................................")
        print("epoch: ", epoch)
        loss = model(feats, pos, mps, nei_index, feat_nei, feat_adj, num_clusters)
        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'HGCC_'+own_str+'.pkl') 
        else:
            cnt_wait += 1
        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        print("best loss: ", best)
        loss.backward()
        optimiser.step()
    model.load_state_dict(torch.load('HGCC_'+own_str+'.pkl'))
    model.eval()
    embeds = model.get_embeds(feats, mps)
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")

    if args.save_emb:
        print("args.save_emb: ")
        print(args.save_emb)
        f = open("embeds/"+args.dataset+"/"+"cluster_random"+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()
    run_kmeans(embeds.cpu(), torch.argmax(label.cpu(), dim=-1), nb_classes, starttime, args.dataset)


if __name__ == '__main__':
    train()

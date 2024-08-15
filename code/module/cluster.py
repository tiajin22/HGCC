import random

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run_kmeans(x, clusters, tau):
    x = x.cpu().detach().numpy()
    results = {'im2cluster': [], 'centroids': [], 'density': []}
    for seed, num_cluster in enumerate(clusters):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = num_cluster
        clus = faiss.Clustering(d, k)
        clus.verbose = False
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.gpu = False
        clus.max_points_per_centroid = 4000
        clus.min_points_per_centroid = 5.
        # res = faiss.StandardGpuResources()
        # cfg = faiss.GpuIndexFlatConfig()
        # cfg.useFloat16 = False
        # cfg.device = 0
        # index = faiss.GpuIndexFlatIP(res, d, cfg)
        index = faiss.IndexFlatIP(d)
        clus.train(x, index)

        D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        im2cluster = []
        for n in I:
            im2cluster.append(n[0])

        count_0 = im2cluster.count(0)
        count_1 = im2cluster.count(1)
        count_2 = im2cluster.count(2)
        im2cluster = np.array(im2cluster)
        np.random.seed(0)

        tsne = TSNE(n_components=2, perplexity=50, random_state=42)  
        x_tsne = tsne.fit_transform(x)

        colors = ['r', 'g', 'b']  
        plt.figure(figsize=(10, 8))

        for i, color in enumerate(colors):
            indices = np.where(np.array(im2cluster) == i)[0]
            plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1], c=color, label=f'Cluster {i}', alpha=0.5)

        # plt.title('Community Distribution Plot using t-SNE')
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.xticks([])  
        plt.yticks([])  

        plt.axis('off')
        plt.savefig('epoch1.png')
        plt.show()
        center = [[] for _ in range(k)]
        centroids = []
        for i in range(len(im2cluster)):
            center[im2cluster[i]].append(x[i])
        for i in range(len(center)):
            random.shuffle(center[i])
            centroids.append(center[i][:int(0.8 * len(center[i])+1)])
        
        result = []
        for i in range(k):
            temp = np.zeros(64)
            for j in range(64):
                if len(centroids[i]):
                    temp[j] = sum(array[j] for array in centroids[i])/len(centroids[i])
            result.append(temp)
        centroids = result

        # centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for _ in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 5)
                density[i] = d

        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  
        density = tau * density / density.mean() 


        centroids = torch.Tensor(centroids).cuda()
        centroids = F.normalize(centroids, p=2, dim=1).cuda()

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results


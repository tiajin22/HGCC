import pickle
import torch as th
import torch.nn.functional as F
import datetime
import pickle as pkl
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from utils import load_data, set_params, evaluate
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from sklearn.cluster import KMeans


if __name__ == '__main__':
    file_path = "embeds/acm/0.pkl"
    # data = pickle.load()
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    estimator = KMeans(n_clusters=3)
    for _ in range(1):
        estimator.fit(data)
        y_pred = estimator.predict(data)
    np.random.seed(0)

    tsne = TSNE(n_components=2, perplexity=50, random_state=42)  
    x_tsne = tsne.fit_transform(data)
    colors = ['r', 'g', 'b']  
    plt.figure(figsize=(10, 8))

    for i, color in enumerate(colors):
        indices = np.where(np.array(y_pred) == i)[0]
        plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1], c=color, label=f'Cluster {i}', alpha=0.5)

    plt.legend()
    plt.xticks([])  
    plt.yticks([])  

    plt.axis('off')
    plt.savefig('my_plot.png')
    plt.show()


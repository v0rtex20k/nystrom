import numpy as np
from typing import *
from numpy.random.mtrand import seed
import  scipy.io as sio
import numpy.linalg as nla
import scipy.linalg as sla
import matplotlib.pyplot as mplt
from sklearn.cluster import KMeans
import scipy.sparse.linalg as ssla
from scipy.sparse import dia_matrix
import matplotlib.colors as mcolors
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist, squareform, pdist

def plot_clustering(data: np.ndarray, true_labels: np.ndarray, 
                    pred_labels: np.ndarray, centers: np.ndarray)-> None:
    utls , upls = np.unique(true_labels), np.unique(pred_labels)
    all_colors = list(mcolors.get_named_colors_mapping().keys())
    (fig , a), cs = mplt.subplots(2), all_colors[:utls.size]

    for ki in range(utls.size):
        p, t = data[pred_labels==upls[ki],:], data[true_labels==utls[ki],:]
        a[0].scatter(t[:,0], t[:,1],c=cs[ki],s=0.5, marker='.')
        a[1].scatter(p[:,0], p[:,1], c=cs[ki],s=0.5, marker='.')
    
    a[0].set_title('Original Clustering')
    a[1].set_title("Nystrom Clustering")
    a[1].scatter(centers[:,0], centers[:,1], c='r',s=25, marker='x')
    # a[0].scatter(data[:,1], data[:,0],c='c',s=5,zorder=-1, marker='.')
    mplt.subplots_adjust(hspace=1, wspace=1); mplt.show()

Cluster = NewType("Relevant stats from KMeans", Tuple[np.ndarray, np.ndarray, float])
def standard_nystrom(X: np.ndarray,  l: int, r: int, k: int, gamma: int = None, seed: int=None)-> Cluster:
    '''
    Standard Nystrom k-Clustering of X

            Parameters:
                    X (ndarray): An (m x n) dataset, where it is assumed m >> n
                    l (int): Number of columns of the affinity matrix sampled
                    r (int): Rank of approximation matrix Wr
                    k (int): Number of expected clusters
                    seed (int): Random seed for sampling - default is None

            Returns:
                    labels (ndarray): k-Clustering label array of shape (m x n)
                    centroids (ndarray): Coordinates of cluster centroids (k x 2)
                    inertia (float): Inertia score of clustering
    '''
    if X.shape[1] > X.shape[0]: X = X.T
    np.random.seed(seed); m = X.shape[0]
    X_sample = X[np.random.choice(m, l, replace=False),:]
    A_hat = rbf_kernel(X_sample, X_sample, gamma=gamma)

    U,E,_ = sla.svd(A_hat, full_matrices=False)

    M = U[:,:k]@np.diag(1/np.sqrt(E[:k]))
    C = rbf_kernel(X, X_sample, gamma=gamma)
    km = KMeans(n_clusters=k, random_state=seed).fit(C@M)
    return (km.labels_.flatten(), km.cluster_centers_.reshape(-1,k), km.inertia_)

def ng_nystrom(X: np.ndarray, k: int, gamma: int = 1, seed: int=None)-> Cluster:
    '''
    Nystrom k-Clustering of X (Ng et al. 2001)

            Parameters:
                    X (ndarray): An (m x n) dataset, where it is assumed m >> n
                    k (int): Number of expected clusters
                    seed (int): Random seed for sampling - default is None

            Returns:
                    labels (ndarray): k-Clustering label array of shape (m x n)
                    centroids (ndarray): Coordinates of cluster centroids (k x 2)
                    inertia (float): Inertia score of clustering
    '''
    if X.shape[1] > X.shape[0]: X = X.T
    np.random.seed(seed); m = X.shape[0]
    d = squareform(pdist(X))
    A_hat = rbf_kernel(d,d,gamma=gamma)
    D = np.diag(1/np.sqrt(np.sum(A_hat, axis=1)))
    vals, vecs = nla.eig(D@A_hat@D)
    Y = vecs[:, np.argsort(vals)[-k:]]
    Y /= np.tile(np.sqrt(np.sum(np.square(Y), axis=1)), (2,1)).T
    km = KMeans(n_clusters=k, random_state=seed).fit(Y)
    return (km.labels_.flatten(), km.cluster_centers_.reshape(-1,k), km.inertia_)


def fast_nystrom(X: np.ndarray,  l: int, r: int, k: int, gamma: int = None, seed: int=None)-> Cluster:
    '''
    Fast Nystrom k-Clustering of X (Choromanska et al. 2013)

            Parameters:
                    X (ndarray): An (m x n) dataset, where it is assumed m >> n
                    l (int): Number of columns of the affinity matrix sampled
                    r (int): Rank of approximation matrix Wr
                    k (int): Number of expected clusters
                    seed (int): Random seed for sampling - default is None

            Returns:
                    labels (ndarray): k-Clustering label array of shape (m x n)
                    centroids (ndarray): Coordinates of cluster centroids (k x 2)
                    inertia (float): Inertia score of clustering
    '''
    if r > l: print("Rank cannot exceed than sample size"); return (None, None, None)
    if X.shape[1] > X.shape[0]: X = X.T
    np.random.seed(seed); m = X.shape[0]
    sample_idxs = np.random.choice(m, l, replace=False)

    X_sample = X[sample_idxs, :]
    A_hat = rbf_kernel(X, X_sample, gamma=gamma)

    D = np.diag(1 / np.sqrt(np.sum(A_hat, axis=1)))
    d = np.diag(1 / np.sqrt(np.sum(A_hat, axis=0)))
    C = (np.eye(m,l) - (np.sqrt(l/m) * (D @ A_hat @ d)))
    W = C[sample_idxs, :]
    UW, EW, VW = nla.svd(W)
    Wr = UW[:,:r]@np.diag(EW[:r])@VW[:r,:]
    EWr, UWr = np.linalg.eig(Wr)
    U_tilde = (np.sqrt(l/m) * C@UWr@nla.inv(np.diag(EWr)))
    UUt, EUt, _ = nla.svd(U_tilde)
    EUt[EUt <= 1e-6] = 0
    Y = np.real(UUt[:, np.argsort(EUt[EUt > 0])[:np.count_nonzero(EUt == 0) + k]])
    Y /= np.tile(np.sqrt(np.sum(np.square(Y), axis=1)), (2,1)).T
    km = KMeans(n_clusters=k, random_state=seed).fit(Y)
    return (km.labels_.flatten(), km.cluster_centers_.reshape(-1,k), km.inertia_)

def freq(v: np.ndarray)->Tuple[Any, float]:
    return np.asarray(np.unique(predicted_labels, return_counts=True)).T

if __name__ == "__main__":
    data, n_clusters = sio.loadmat('Moon.mat')['x'], 2
    original_labels = sio.loadmat('Moon_Label.mat')['y'].flatten()
    sample_size, expected_rank = 275, 275
    # predicted_labels, centroids, I = ng_nystrom(data, n_clusters, seed=17)
    predicted_labels, centroids, I = fast_nystrom(data, sample_size, expected_rank, n_clusters, seed=17)

    freq(predicted_labels)

    plot_clustering(data, original_labels, predicted_labels, centroids)
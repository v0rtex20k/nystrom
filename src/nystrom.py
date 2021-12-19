import os
import numpy as np
import pandas as pd
import  scipy.io as sio
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse.dia import dia_matrix
from scipy.spatial.distance import squareform, pdist

import matplotlib.pyplot as mplt

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import *
from random import shuffle
from typing import *

NUM_CORES = os.cpu_count()
EPSILON   = 1e-10 # avoid DivideByZero errors


Cluster = NewType("Relevant stats from MiniBatchKMeans", Tuple[np.ndarray, np.ndarray, float])
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
    global EPSILON
    if X.shape[1] > X.shape[0]: X = X.T
    np.random.seed(seed); m = X.shape[0]
    X_sample = X[np.random.choice(m, l, replace=False),:]
    A_hat = rbf_kernel(X_sample, X_sample, gamma=gamma)

    U,E,_ = sla.svd(A_hat, full_matrices=False)

    M = U[:,:k]@np.diag(1/(np.sqrt(E[:k]) + EPSILON))
    C = rbf_kernel(X, X_sample, gamma=gamma)
    mbk = MiniBatchKMeans(n_clusters=k, random_state=seed).fit(C@M)
    return mbk.labels_.flatten() # , mbk.cluster_centers_.reshape(-1,k), mbk.inertia_)

# import asyncio
# def background(f):
#     def wrapped(*args, **kwargs):
#         runnable = partial(f, *args, **kwargs)
#         return asyncio.get_event_loop().run_in_executor(None, runnable)

#     return wrapped

# @background
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs
def sparse_ng_nystrom(X: np.ndarray, k: int, gamma: int = None, seed: int=None)-> np.ndarray:
    '''
    SPARSE Nystrom k-Clustering of X (Ng et al. 2001)

            Parameters:
                    X (ndarray): An (m x n) dataset, where it is assumed m >> n
                    k (int): Number of expected clusters
                    seed (int): Random seed for sampling - default is None

            Returns:
                    labels (ndarray): k-Clustering label array of shape (m x n)
    '''
    global EPSILON
    if X.shape[1] > X.shape[0]: X = X.T
    np.random.seed(seed)
    d = squareform(pdist(X))
    A_hat =  rbf_kernel(d,d,gamma=gamma)
    dvals: np.ndarray = 1/(np.sqrt(np.sum(A_hat, axis=1)) + EPSILON)
    D: dia_matrix = sparse.dia_matrix((dvals.T, np.zeros(1)), shape=(dvals.size, dvals.size))
    vecs = np.real(eigs((D*A_hat*D), k=k, which='LM')[1])
    Y = vecs / (np.tile(np.sqrt(np.sum(np.square(vecs), axis=1)), (k,1)).T + EPSILON)
    mbk = SpectralClustering(n_clusters=k, random_state=seed, n_jobs=-1, degree=17, affinity="polynomial").fit(Y) 
    return mbk.labels_.flatten()

def ng_nystrom(X: np.ndarray, k: int, gamma: int = None, seed: int=None)-> Cluster:
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
    global EPSILON
    if X.shape[1] > X.shape[0]: X = X.T
    np.random.seed(seed)
    d = squareform(pdist(X))
    A_hat = rbf_kernel(d,d,gamma=gamma)
    D = np.diag(1/(np.sqrt(np.sum(A_hat, axis=1)) + EPSILON))
    vals, vecs = nla.eig(D@A_hat@D)
    Y = np.real(vecs[:, np.argsort(vals)[-k:]])
    Y /= np.tile(np.sqrt(np.sum(np.square(Y), axis=1)), (k,1)).T + EPSILON
    mbk = MiniBatchKMeans(n_clusters=k, random_state=seed).fit(Y)
    return mbk.labels_.flatten() # , mbk.cluster_centers_.reshape(-1,k), mbk.inertia_)

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

    # I think something is wrong here,
    # but not sure exactly what. Focus on slides,
    # then come back to try to fix

    global EPSILON
    if r > l: print("Rank cannot exceed than sample size"); return None
    if X.shape[1] > X.shape[0]: X = X.T
    np.random.seed(seed); m,n = X.shape
    sample_idxs = np.random.choice(n, l, replace=False)

    dists = squareform(pdist(X[:, sample_idxs]))
    A_hat = rbf_kernel(dists, dists, gamma=gamma)

    D = np.diag(1 / (np.sqrt(np.sum(A_hat, axis=1)) + EPSILON))
    d = np.diag(1 / (np.sqrt(np.sum(A_hat, axis=0)) + EPSILON))
    C = (np.eye(m,l) - (np.sqrt(l/m) * (D @ A_hat @ d))[:,:l]) # (m x l)
    W = C[sample_idxs, :]
    UW, EW, VW = nla.svd(W)
    Wr = UW[:,:r]@np.diag(EW[:r])@VW[:r,:]
    EWr, UWr = nla.eig(Wr.T@Wr)
    print(C.shape, UWr.shape, nla.pinv(np.diag(EWr)).shape)
    U_tilde = (np.sqrt(l/m) * C@UWr@nla.pinv(np.diag(EWr)))
    EUt, UUt = nla.eig(U_tilde.T@U_tilde)
    Y = np.real(UUt[:, np.argsort(EUt)[-k:]]) # np.argsort(EUt[EUt > 0])[:np.count_nonzero(EUt == 0) + k]])
    Y /= np.tile(np.sqrt(np.sum(np.square(Y), axis=1)), (k,1)).T + EPSILON
    # mbk = MiniBatchKMeans(n_clusters=k, random_state=seed).fit(Y)
    mbk = SpectralClustering(n_clusters=k, random_state=seed, n_jobs=-1).fit(Y) 
    return mbk.labels_.flatten() # , mbk.cluster_centers_.reshape(-1,k), mbk.inertia_)

def freq(v: np.ndarray)->Tuple[Any, float]: # utility function
    return np.asarray(np.unique(v, return_counts=True)).T

def load_sample_data(name: str, subsample: int=None)-> Dict[str, Any]:
    try: 
        data_sources = {
            "moon": {"data": sio.loadmat('../data/Moon.mat')['x'],
                    "labels": sio.loadmat('../data/Moon_Label.mat')['y'].flatten(),
                    "n_clusters": 2, "sample_size": 275, "expected_rank": 275
                    },
            "circ": {"data": sio.loadmat('../data/concentric_circles_label.mat')['X'],
                    "labels": sio.loadmat('../data/concentric_circles.mat')['labels'].flatten(),
                    "n_clusters": 3, "sample_size": 25, "expected_rank": 25
                    },
            "cali": {"data": pd.read_csv("../data/housing.csv").loc[:, ["Latitude", "Longitude"]].to_numpy(),
                    "labels": MiniBatchKMeans(n_clusters=6).fit_predict(pd.read_csv("../data/housing.csv").loc[:, ["Latitude", "Longitude"]].to_numpy()),
                    "n_clusters": 6, "sample_size": 1000, "expected_rank": 1000
                    }
        }
        name = name.lower()
        if name in data_sources.keys():
            data = data_sources[name]
            print(f'Loading \"{name}\" dataset {data["data"].shape}')
            if subsample is not None:
                np.random.seed(19); m = data["data"].shape[0]
                sample_idxs = np.random.choice(m, subsample, replace=False)
                data["data"] = data["data"][sample_idxs,:]
                data["labels"] = data["labels"][sample_idxs]
            return data
        raise FileNotFoundError

    except FileNotFoundError: print(f"Could not find \"{name}\" dataset"); exit()


if __name__ == "__main__":
        data = load_sample_data("cali", subsample=5000)

        import test__algos

        # test__algos.cluster1(data["data"], data["labels"], data["n_clusters"],
        #                      frac_train=0.75, split_seed=42, fit_seed=17, verbose=True)

        test__algos.cluster0(data["data"], data["labels"], data["n_clusters"],
                frac_train=0.75, split_seed=42, fit_seed=17, verbose=True)

        test__algos.cluster1(data["data"], data["labels"], data["n_clusters"],
                frac_train=0.75, split_seed=42, fit_seed=17, verbose=True)

        test__algos.cluster3(data["data"], 500, 500, data["labels"], data["n_clusters"],
                frac_train=0.75, split_seed=42, fit_seed=17, verbose=True)

        # predicted_labels = ng_nystrom(data["data"], data["n_clusters"])
        # predicted_labels, centroids, I = standard_nystrom(data["data"], data["sample_size"], data["expected_rank"], data["n_clusters"], seed=17)
        # predicted_labels = fast_nystrom(data["data"], data["sample_size"], data["expected_rank"], data["n_clusters"], seed=17)
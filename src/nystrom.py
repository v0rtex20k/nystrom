import numpy as np
from typing import *
import numpy.linalg as nla
import scipy.sparse.linalg as ssla
from scipy.sparse import dia_matrix
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

Cluster = NewType("Relevant stats from KMeans", Tuple[np.ndarray, np.ndarray, np.ndarray])
def nystrom_cluster(X: np.ndarray,  l: int, r: int, k: int, seed: int=None)-> Cluster:
    '''
    Returns a Fast Nystrom k-Clustering of a 2D Dataset (Choromanska et al. 2013)

            Parameters:
                    X (ndarray): An (m x n) dataset, where it is assumed m >> n
                    l (int): Number of columns of the Affinity Matrix sampled
                    r (int): Rank of approximation matrix Wr
                    k (int): Number of expected clusters
                    seed (int): Random seed for sampling - default is None

            Returns:
                    labels (ndarray): k-Clustering label array of shape (m x n)
    '''
    if X.shape[1] > X.shape[0]: X = X.T
    np.random.seed(seed); m = X.shape[0]
    sample_idxs = np.random.choice(m, l, replace=False)
    A_hat = cdist(X, X[sample_idxs, :]) # can we force A_hat to be sparse somehow?
    D = 1 / np.sqrt(np.sum(A_hat, axis=1))
    d = 1 / np.sqrt(np.sum(A_hat, axis=0))
    C = (np.eye(m,l) - (np.sqrt(l/m) * (D @ A_hat @ d)))
    Ur, Er, Vr = ssla.svds(dia_matrix(C[sample_idxs,:]), k=r)
    Ur = (np.sqrt(l/m) * C@Ur@nla.pinv(np.diag(Er))); Er *= m/l
    Y = Ur[:, np.count_nonzero(Er == 0) + np.argsort(Er[Er > 0])[:k]]
    Y /= np.sqrt(np.sum(np.square(Y), axis=0))
    clustering = KMeans(n_clusters=k, random_state=seed).fit(Y)
    return (clustering.labels_, clustering.cluster_centers_, clustering.inertia_)

fake_data = np.random.rand(500,2)
L,C,I = nystrom_cluster(fake_data, 50, 10, 3)
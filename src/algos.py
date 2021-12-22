import sys
import numpy as np
from typing import *
import scipy.linalg as sla
import scipy.sparse as sparse
import jax.numpy.linalg as nla
import matplotlib.pyplot as mplt
import scipy.sparse.linalg as ssl
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from scipy.sparse.csr import csr_matrix
from scipy.sparse.dia import dia_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import squareform, pdist

EPSILON = 1e-10 # avoid DivideByZero errors
MEM_PREC = 3    # for memory profiling output
import warnings; warnings.filterwarnings('ignore')

def algo0(S: np.ndarray, k: int, seed: int=None)-> Tuple[np.ndarray, int]:
    '''
        Params:
            - S : {s1, s2, ..., sN}, where si ∊ R^d ==> S ∊ R^(n x d)
            - k : number of clusters
            - action: {'f': KMeans.fit(), 'p': KMeans.predict()}
            - preds: precomputed classifier for predictions, ignored if action != 'p'
            - seed: random seed for sampling - default is None
        Returns:
            - preds: predicted label array
    '''
    return KMeans(n_clusters=k, random_state=seed).fit_predict(np.real(S)), sys.getsizeof(S)

def algo1(S: np.ndarray, k: int, style: str='ng', seed: int=None)-> Tuple[np.ndarray, int]:
    '''
        Params:
            - S : {s1, s2, ..., sN}, where si ∊ R^d ==> S ∊ R^(n x d)
            - k : number of clusters
            - action: {'f': KMeans.fit(), 'p': KMeans.predict()}
            - preds: precomputed classifier for predictions, ignored if action != 'p'
            - style: {'ng': Ng et. al 2001, 'cho': Choromanska et al. 2013}
            - seed: random seed for sampling - default is None
        Returns:
            - preds: predicted label array
    '''
    X: csr_matrix; L: np.ndarray; global EPSILON
    A: np.ndarray = rbf_kernel(squareform(pdist(S))) # A ∊ R^(n x n)
    np.fill_diagonal(A,0)
    dvals: np.ndarray = 1/(np.sqrt(np.sum(A, axis=1)) + EPSILON)
    d: dia_matrix = sparse.dia_matrix(np.diag(dvals)) # d ∊ R^(n x n)       

    if style.lower() == "ng":
        L = d@A@d                               # L ∊ R^(n x n), unnormalized
        vals, vecs = nla.eig(L)
        X = vecs[:,np.argsort(vals)[-k:]]       # X ∊ R^(n x k) --> k LARGEST eigenvectors
        
    elif style.lower() == "cho":
        L = np.eye(S.shape[0]) - d@A@d          # L ∊ R^(n x n), normalized
        vals, vecs = nla.eig(L)
        X = vecs[:,np.argsort(vals)[-k:]]     # X ∊ R^(n x k) --> k SMALLEST eigenvectors

    return KMeans(n_clusters=k,random_state=seed).fit_predict(np.real(X)), sys.getsizeof(A)

def algo3(S: np.ndarray, l: int, r: int,  k: int, seed: int=None)-> Tuple[np.ndarray, int]:
    '''
        Params:
            - S : {s1, s2, ..., sN}, where si ∊ R^d ==> S ∊ R^(n x d)
            - k : number of clusters
            - l : number of columns sampled
            - r : rank approximation
            - action: {'f': KMeans.fit(), 'p': KMeans.predict()}
            - preds: precomputed classifier for predictions, ignored if action != 'p'
            - style: {'ng': Ng et. al 2001, 'cho': Choromanska et al. 2013}
            - seed: random seed for sampling - default is None
        Returns:
            - preds: predicted label array
    '''
    global EPSILON; h = np.sqrt(l/S.shape[0]); p = min(l, S.shape[1])
    if not k <= r <= l <= S.shape[0]: exit("Rank cannot be larger than column sample size")
    col_idxs: np.ndarray = np.random.choice(S.shape[1], p, replace=False)
    A: np.ndarray = rbf_kernel(S)[:,col_idxs] # A ∊ R^(n x l)
    np.fill_diagonal(A,0)

    Dvals: np.ndarray = 1/(np.sqrt(np.sum(A, axis=1)) + EPSILON)
    dvals: np.ndarray = 1/(np.sqrt(np.sum(A, axis=0)) + EPSILON)
    D: dia_matrix = sparse.dia_matrix((Dvals.T, [0]), shape=(Dvals.size, Dvals.size))
    d: dia_matrix = sparse.dia_matrix((dvals.T, [0]), shape=(dvals.size, dvals.size))

    C: np.ndarray = sparse.csr_matrix(np.eye(S.shape[0],p)) - h * D@A@d # C ∊ R^(n x l), not sparse
    W: np.ndarray = C[col_idxs,:] # W ∊ R^(l x l), not sparse
    EW, UW = sla.eig(W) # EW ∊ R^(l x l), UW ∊ R^(l x l)
    
    r_idxs: np.ndarray = np.argsort(EW)[:r]
    UWr: dia_matrix = sparse.dia_matrix(UW[:, r_idxs]) # UWr ∊ R^(l x r)
    EWr: dia_matrix = sparse.dia_matrix(np.diag(EW[r_idxs])) # EWr ∊ R^(r x r)
    Ut: np.ndarray = h * C@UWr@ssl.inv(EWr) # Ut ∊ R^(n x r)
    Y = sla.svd(Ut)[0][:,-k:] # vh ∊ R^(n x k) --> top k *unitary* eigenvectors
    return KMeans(n_clusters=k, random_state=seed).fit_predict(np.real(Y)), sys.getsizeof(A)

def plot_clustering(X: np.ndarray, true_labels: np.ndarray, pred_labels: np.ndarray)-> None:
    '''
        Utility function: Plot predictions after clustering

        Params
            - X: Dataset to be plotted
            - true_labels: Actual label array
            - pred_labels: Predicted label array
    '''
    utls, upls = np.unique(true_labels), np.unique(pred_labels)
    (_,a), cs = mplt.subplots(2), shuffle(['b','g','r','c','m','y','k','w'])
    if upls.shape != utls.shape: upls = utls
    print(X.shape, pred_labels.shape, true_labels.shape, upls.shape, utls.shape)
    for ki in range(utls.size):
        p, t = X[pred_labels==upls[ki],:], X[true_labels==utls[ki],:]
        a[0].scatter(t[:,0], t[:,1],c=cs[ki],s=5, marker='.')
        a[1].scatter(p[:,0], p[:,1], c=cs[ki],s=5, marker='.')
    
    a[0].set_title('Original Clustering')
    a[1].set_title("Nystrom Clustering")
    mplt.subplots_adjust(hspace=1, wspace=1); mplt.show()

def cluster0(data: np.ndarray, labels: np.ndarray=None, n_clusters: int=None, 
             fit_seed: int=None, score_seed: int=None)-> Tuple[np.ndarray, np.ndarray]:
    
    if labels is None: labels = np.zeros(data.shape[0])
    if n_clusters is None: n_clusters = np.unique(labels).size

    preds, size = algo0(data, n_clusters, seed=fit_seed)
    score = silhouette_score(data, preds, metric='euclidean', random_state=score_seed)
    return preds, score, size

def cluster1(data: np.ndarray, labels: np.ndarray=None, n_clusters: int=None,
             fit_seed: int=None, score_seed: int=None)-> Tuple[np.ndarray, np.ndarray]:
    
    if labels is None: labels = np.zeros(data.shape[0])
    if n_clusters is None: n_clusters = np.unique(labels).size

    preds, size = algo1(data, n_clusters, style='ng', seed=fit_seed)
    score = silhouette_score(data, preds, metric='euclidean', random_state=score_seed)
    return preds, score, size

def cluster3(data: np.ndarray, l: int, r: int, labels: np.ndarray=None,
             n_clusters: int=None, fit_seed: int=None, score_seed: int=None)-> Tuple[np.ndarray, np.ndarray]:
    
    if labels is None: labels = np.zeros(data.shape[0])
    if n_clusters is None: n_clusters = np.unique(labels).size

    preds, size = algo3(data, l, r, n_clusters, seed=fit_seed)
    score = silhouette_score(data, preds, metric='euclidean', random_state=score_seed)
    return preds, score, size
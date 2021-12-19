import os
import numpy as np
from typing import *
import scipy as sp
import scipy.sparse as sparse
from numpy import linalg as nla, ndarray
import matplotlib.pyplot as mplt
from scipy.sparse.csc import csc_matrix
import scipy.sparse.linalg as ssl
from sklearn.utils import shuffle
from scipy.sparse.csr import csr_matrix
from scipy.sparse.dia import dia_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import MiniBatchKMeans as MBK
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import squareform, pdist

NUM_CORES = os.cpu_count()
EPSILON = 1e-10 # avoid DivideByZero errors

import warnings; warnings.filterwarnings('ignore')

def algo0(S: np.ndarray, k: int, action: str, clf: MBK=None, seed: int=None)-> Union[MBK, np.ndarray]:
    '''
        Params:
            - S : {s1, s2, ..., sN}, where si ∊ R^d ==> S ∊ R^(n x d)
            - k : number of clusters
            - action: {'f': KMeans.fit(), 'p': KMeans.predict()}
            - clf: precomputed classifier for predictions, ignored if action != 'p'
            - seed: random seed for sampling - default is None
        Returns:
            - Fitted MultiBatchKMeans Classifier | predicted label array
    '''
    if action.lower() == "f" and clf is None:
        #return MBK(n_clusters=k,batch_size=300*NUM_CORES,random_state=seed).fit(np.real(S))
        return KMeans(n_clusters=k, random_state=seed).fit(np.real(S))
    elif action.lower() == "p" and clf is not None: 
        return clf.predict(np.real(S))
    else: exit(f"Invalid action {action}")

def algo1(S: np.ndarray, k: int, action: str, clf: MBK=None, style: str='ng', seed: int=None)-> Union[MBK, np.ndarray]:
    '''
        Params:
            - S : {s1, s2, ..., sN}, where si ∊ R^d ==> S ∊ R^(n x d)
            - k : number of clusters
            - action: {'f': KMeans.fit(), 'p': KMeans.predict()}
            - clf: precomputed classifier for predictions, ignored if action != 'p'
            - style: {'ng': Ng et. al 2001, 'cho': Choromanska et al. 2013}
            - seed: random seed for sampling - default is None
        Returns:
            - Fitted MultiBatchKMeans Classifier | predicted label array
    '''
    X: csr_matrix; global EPSILON
    A: np.ndarray = rbf_kernel(S) # A ∊ R^(n x n)
    np.fill_diagonal(A,0)
    dvals: np.ndarray = 1/(np.sqrt(np.sum(A, axis=1)) + EPSILON)
    d: dia_matrix = sparse.dia_matrix((dvals.T, np.zeros(1)), shape=(dvals.size, dvals.size)) # d ∊ R^(n x n)       

    if style.lower() == "ng":
        L: np.ndarray = d@A@d                          # L ∊ R^(n x n), unnormalized
        X = np.real(ssl.eigs((L), k=k, which='LM')[1]) # X ∊ R^(n x k) --> k LARGEST eigenvectors
    elif style.lower() == "cho":
        L: np.ndarray = np.eye(S.shape[0]) - d@A@d     # L ∊ R^(n x n), normalized
        X = np.real(ssl.eigs((L), k=k, which='SM')[1]) # X ∊ R^(n x k) --> k SMALLEST eigenvectors

    if action.lower() == "f" and clf is None:
        return MBK(n_clusters=k,batch_size=300*NUM_CORES,random_state=seed).fit(np.real(X))
    elif action.lower() == "p" and clf is not None: 
        return clf.predict(np.real(X))
    else: exit(f"Invalid action {action}")

def algo3(S: np.ndarray, l: int, r: int,  k: int, action: str, clf: MBK=None, seed: int=None)-> Union[MBK, np.ndarray]:
    '''
        Params:
            - S : {s1, s2, ..., sN}, where si ∊ R^d ==> S ∊ R^(n x d)
            - k : number of clusters
            - l : number of columns sampled
            - r : rank approximation
            - action: {'f': KMeans.fit(), 'p': KMeans.predict()}
            - clf: precomputed classifier for predictions, ignored if action != 'p'
            - style: {'ng': Ng et. al 2001, 'cho': Choromanska et al. 2013}
            - seed: random seed for sampling - default is None
        Returns:
            - Fitted MultiBatchKMeans Classifier | predicted label array
    '''
    X: csr_matrix; global EPSILON
    n = S.shape[0]
    h = np.sqrt(l/n)
    if not k <= r <= l <= n: exit("Rank cannot be larger than column sample size")
    A: np.ndarray = rbf_kernel(squareform(pdist(S))) # A ∊ R^(n x l)
    col_idxs: np.ndarray = np.random.choice(A.shape[1], l, replace=False)
    A: np.ndarray = A[:,col_idxs]; np.fill_diagonal(A,0)
    Dvals: np.ndarray = 1/(np.sqrt(np.sum(A, axis=1)) + EPSILON)
    dvals: np.ndarray = 1/(np.sqrt(np.sum(A, axis=0)) + EPSILON)
    D: csc_matrix = sparse.csc_matrix(np.diag(Dvals)) # D ∊ R^(n x n)
    d: csc_matrix = sparse.csc_matrix(np.diag(dvals)) # d ∊ R^(l x l)

    C: np.ndarray = sparse.csc_matrix(np.eye(n,l)) - h * D@A@d # C ∊ R^(n x l), not sparse
    W: np.ndarray = C[col_idxs,:] # W ∊ R^(l x l), not sparse
    EW, UW = nla.eig(W) # EW ∊ R^(l x l), UW ∊ R^(l x l)
    
    r_idxs: np.ndarray = np.argsort(EW)[:r]
    UWr: dia_matrix = sparse.dia_matrix(UW[:, r_idxs]) # UWr ∊ R^(l x r)
    EWr: csc_matrix = sparse.csc_matrix(np.diag(EW[r_idxs])) # EWr ∊ R^(r x r)

    Ut: np.ndarray = h * C@UWr@ssl.inv(EWr) # Ut ∊ R^(n x r)
    u = np.real(nla.svd(Ut)[0]) # vh ∊ R^(n x k) --> k SMALLEST *unitary* eigenvectors
    Y: np.ndarray = u[:,:k]
    
    #print(A.shape, D.shape, d.shape, C.shape, W.shape, UWr.shape, EWr.shape, Ut.shape, vh.shape)
    # exit("\n\t========< EXIT >========\n")

    if action.lower() == "f" and clf is None:
        return MBK(n_clusters=k,batch_size=300*NUM_CORES,random_state=seed).fit(np.real(Y))
    elif action.lower() == "p" and clf is not None: 
        return clf.predict(np.real(Y))
    else: exit(f"Invalid action {action}")

def plot_clustering(X: np.ndarray, true_labels: np.ndarray, pred_labels: np.ndarray)-> None:
    '''
        Plot predictions after clustering

        Params
            - X: Dataset to be plotted
            - true_labels: Actual label array
            - pred_labels: Predicted label array
    '''
    utls, upls = np.unique(true_labels), np.unique(pred_labels)
    (fig , a), cs = mplt.subplots(2), shuffle(['b','g','r','c','m','y','k','w'])
    if upls.shape != utls.shape: upls = utls
    print(X.shape, pred_labels.shape, true_labels.shape, upls.shape, utls.shape)
    for ki in range(utls.size):
        p, t = X[pred_labels==upls[ki],:], X[true_labels==utls[ki],:]
        a[0].scatter(t[:,0], t[:,1],c=cs[ki],s=5, marker='.')
        a[1].scatter(p[:,0], p[:,1], c=cs[ki],s=5, marker='.')
    
    a[0].set_title('Original Clustering')
    a[1].set_title("Nystrom Clustering")
    mplt.subplots_adjust(hspace=1, wspace=1); mplt.show()

def label_counts(l: np.ndarray)-> np.ndarray:
    '''
        Counts the number of observations assigned to each cluster

        Params
            - l: label array ∊ R^(n)
        Returns
            - c: count array ∊ R^(n x 2)
    '''
    return np.asarray(np.unique(l, return_counts=True)).T


from sklearn.metrics import accuracy_score, silhouette_score

def cluster0(data: np.ndarray, labels: np.ndarray=None, n_clusters: int=None,
frac_train: float=0.75, split_seed: int=None, fit_seed: int=None, verbose: bool=True)-> Tuple[np.ndarray, np.ndarray]:
    
    if labels is None: labels = np.zeros(data.shape[0])
    if n_clusters is None: n_clusters = np.unique(labels).size

    if verbose: print("Splitting...")
    X_train, X_test, _, y_test = train_test_split(data, labels, train_size=frac_train, random_state=split_seed)
    if verbose: print("Fitting...")
    clf = algo0(X_train, n_clusters, action='f', seed=fit_seed)
    if verbose: print("Predicting...")
    y_pred: np.ndarray = algo0(X_test, n_clusters, action='p', clf=clf)
    if verbose: print("TRUE:\n",label_counts(y_test))
    if verbose: print("PREDICTED:\n",label_counts(y_pred))
    #if verbose: print("Plotting...")
    #plot_clustering(X_test, y_test, predicted_labels)


    print('SCORE: ', silhouette_score(X_test, y_pred, metric='euclidean'))

    return y_test, y_pred

def cluster1(data: np.ndarray, labels: np.ndarray=None, n_clusters: int=None,
frac_train: float=0.75, split_seed: int=None, fit_seed: int=None, verbose: bool=True)-> Tuple[np.ndarray, np.ndarray]:
    
    if labels is None: labels = np.zeros(data.shape[0])
    if n_clusters is None: n_clusters = np.unique(labels).size

    if verbose: print("Splitting...")
    X_train, X_test, _, y_test = train_test_split(data, labels, train_size=frac_train, random_state=split_seed)
    if verbose: print("Fitting...")
    clf = algo1(X_train, n_clusters, action='f', style='cho', seed=fit_seed)
    if verbose: print("Predicting...")
    y_pred: np.ndarray = algo1(X_test, n_clusters, action='p', style='cho', clf=clf)
    if verbose: print("TRUE:\n",label_counts(y_test))
    if verbose: print("PREDICTED:\n",label_counts(y_pred))
    #if verbose: print("Plotting...")
    #plot_clustering(X_test, y_test, predicted_labels)

    print('SCORE: ', silhouette_score(X_test, y_pred, metric='euclidean'))
    return y_test, y_pred

def cluster3(data: np.ndarray, l: int, r: int, labels: np.ndarray=None, n_clusters: int=None,
frac_train: float=0.75, split_seed: int=None, fit_seed: int=None, verbose: bool=True)-> Tuple[np.ndarray, np.ndarray]:
    
    if labels is None: labels = np.zeros(data.shape[0])
    if n_clusters is None: n_clusters = np.unique(labels).size

    if verbose: print("Splitting...")
    X_train, X_test, _, y_test = train_test_split(data, labels, train_size=frac_train, random_state=split_seed)
    if verbose: print("Fitting...")
    clf = algo3(X_train, l, r, n_clusters, action='f', seed=fit_seed)
    if verbose: print("Predicting...")
    y_pred: np.ndarray = algo3(X_test, l, r, n_clusters, action='p', clf=clf)
    if verbose: print("TRUE:\n",label_counts(y_test))
    if verbose: print("PREDICTED:\n",label_counts(y_pred))

    print('SCORE:', silhouette_score(X_test, y_pred, metric='euclidean'))
    return y_test, y_pred
    #print("Plotting...")
    #plot_clustering(X_test, y_test, predicted_labels)
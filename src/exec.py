from functools import partial
import json
from math import ceil
import re
from numpy.random import gamma, seed
import pywt
import numpy as np
import pandas as pd
from typing import *
import matplotlib.pyplot as mplt
from sklearn.utils import Bunch
from nystrom import *
from sklearn.datasets import fetch_openml

import dask.array as da
import dask.bag as db

from timeit import default_timer as timer
from datetime import datetime as dt
from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import mutual_info_classif

def compress(X: np.ndarray, sample_size: int=None, wavelet_name: str='haar', seed: int=None)-> Tuple[np.ndarray, np.ndarray]:
    '''
        Samples and compresses the data using wavelets to speed up clustering

        Params:
            - X: The input array of shape (m,n)
            - sample_size: How many rows to sample from X -> defaults to all rows
            - wavelet_name: Which wavelet to use (see pywt.families() for full list) -> defaults to haar
            - seed: rng seed for reproducibility -> defaults to None (i.e completely random)
        Returns:
            - wavy_sample: The sparsified multi-level wavelet decomposition of the sample of X -
                           should be of size (sample_size, pywt.dwt_max_level(n, 'haar'))
            - idxs: The sampled row indices ()

        For more info on wavedec: http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/wavelet/ch02_u15.html
    '''
    m,n = X.shape
    if m < n: X = X.T
    np.random.seed(seed)
    idxs = np.random.choice(m, m if sample_size is None else sample_size)
    return np.apply_along_axis(lambda r: pywt.wavedec(r, wavelet_name)[0].flatten(), 1, X[idxs,:]), idxs.flatten()

def decompress(coeffs: np.ndarray, wavelet_name: str='haar')-> np.ndarray:
    '''
        Decompresses the data using wavelets to retrieve original images

        Params:
            - X: The input array of shape (m,n)
            - sample_size: How many rows to sample from X -> defaults to all rows
            - wavelet_name: Which wavelet to use (see pywt.families() for full list) -> defaults to haar
            - seed: rng seed for reproducibility -> defaults to None (i.e completely random)
        Returns:
            - wavy_sample: The sparsified multi-level wavelet decomposition of the sample of X -
                           should be of size (sample_size, pywt.dwt_max_level(n, 'haar'))
            - idxs: The sampled row indices ()

        For more info on wavedec: http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/wavelet/ch02_u15.html
    '''
    return np.apply_along_axis(lambda r: pywt.waverec2(r.reshape(28,28), wavelet_name)[0], 1, coeffs)

def sample(X: np.ndarray, sample_size: int=1000, seed:int = None)-> np.ndarray:
    '''
        Samples and compresses the data using simple one-hot encodings to speed up clustering

        Params:
            - X: The input array of shape (m,n)
            - sample_size: How many rows to sample from X -> defaults to all rows
            - wavelet_name: Which wavelet to use (see pywt.families() for full list) -> defaults to haar
            - seed: rng seed for reproducibility -> defaults to None (i.e completely random)
        Returns:
            - wavy_sample: The sparsified multi-level wavelet decomposition of the sample of X -
                           should be of size (sample_size, pywt.dwt_max_level(n, 'haar'))
            - idxs: The sampled row indices ()

        For more info on wavedec: http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/wavelet/ch02_u15.html
    '''
    m,n = X.shape
    if m < n: X = X.T
    np.random.seed(seed)
    idxs = np.random.choice(m, m if sample_size is None else sample_size)
    return X[idxs,:], idxs.flatten()
    # return np.apply_along_axis(lambda r: np.argwhere(r==1).flatten(), 1, X[idxs,:]), idxs.flatten()


def accuracy(preds: np.ndarray, trues: np.ndarray)-> float:
    diffs = np.subtract(preds.flatten(), trues.flatten()[sample_idxs])
    return 100 * np.count_nonzero(diffs == 0) / diffs.size
    
def counts(preds: np.ndarray, trues: np.ndarray, idxs: np.ndarray, nbins: int=10)-> float:
    bin_dict = {}
    for i in range(nbins): # true label of 0
        bin_idxs = np.argwhere(trues.flatten()[idxs].flatten() == i).flatten()
        cdict = {}
        for j in range(nbins):
            cdict[j] = np.count_nonzero(preds[bin_idxs] == j)
        bin_dict[i] = cdict
    
    for k,v in bin_dict.items():
        tot = sum(v.values())
        if tot == 0: continue
        for l,w in v.items():
            bin_dict[k][l] /= tot

    return bin_dict

def most_important_pixels(data: np.ndarray, labels: np.ndarray, pkeep: float=0.1, seed:int = None)-> np.ndarray:
    n, mi = data.shape[1], mutual_info_classif(data, labels, random_state=seed)
    mask = pd.Series(mi, np.arange(n)).fillna(0).rank(ascending=False)
    mask = mask.replace(mask.max(),n).to_numpy()
    ranks, cutoff = np.argsort(mask), int(abs(pkeep)*n) if 0 <= abs(pkeep) <= 1 else int(abs(pkeep))
    mask[ranks[cutoff:]], mask[ranks[:cutoff]] = False, True
    return mask.astype(bool).flatten()

dur = lambda a,b: timedelta(seconds=b-a)
if __name__ == "__main__":
    print('\tLoading MNIST...')
    mnist_bunch: Bunch = fetch_openml(name='mnist_784', version='1')
    data, true_labels = [mnist_bunch[c].to_numpy(dtype=int, na_value=0) for c in ['data', 'target']] # data.shape == (70000, 784)
    print('\tTraining')
    training_data, training_idxs = sample(data, sample_size=20000, seed=17)
    testing_mask = np.ones(data.shape[0], dtype=bool)
    testing_mask[training_idxs] = False
    tick = timer()
    mip_mask = most_important_pixels(training_data, true_labels[training_idxs].flatten(), pkeep=0.5)
    tock = timer()
    print(f"MIP MASK COMPUTED IN {dur(tick, tock).seconds} SECONDS")

    print(testing_mask.ndim, mip_mask.ndim)

    test_grid = np.ix_(testing_mask.flatten(), mip_mask.flatten())
    print([g.shape for g in test_grid])
    testing_data = data[test_grid]
    testing_labels, nbins = true_labels[testing_mask].flatten(), 10
    
    from test__algos import *

    testing_data = testing_data / 255
    # for n in [500, 1000, 2000]:
    #     test_sample, sample_idxs = sample(testing_data, sample_size=n, seed=17)
    #     print(f'\tClustering MNIST w/ {n}...')
    #     tick = timer()
    #     #pred_labels = sparse_ng_nystrom(test_sample, k=nbins, gamma=None, seed=17)
    #     pred_labels = fast_nystrom(test_sample, 60, 50, nbins, seed=17)
    #     tock  = timer()
    #     print(f"NYSTROM w/ {pred_labels.shape} COMPLETED IN {dur(tick, tock).seconds} SECONDS")

    for n in [500, 1000, 2000]:
        test_sample, sample_idxs = sample(testing_data, sample_size=n, seed=19)
        print(f'\tClustering MNIST w/ {n}...')
        tick = timer()
        # cluster_test_labels, pred_labels = cluster1(test_sample, testing_labels[sample_idxs],
        #                                  nbins, frac_train=0.5, split_seed=42, fit_seed=17)

        print(test_sample.shape)
        cluster_test_labels, pred_labels = cluster3(test_sample, test_sample.shape[1], test_sample.shape[1], testing_labels[sample_idxs], nbins, 
                                                    frac_train=0.5, split_seed=71, fit_seed=17, verbose=True)
        tock  = timer()
        print(f"NYSTROM w/ {pred_labels.shape} COMPLETED IN {dur(tick, tock).seconds} SECONDS")
        # print('\tAssessing performance and saving results...')
        # with open(f'flipped-FAST-ALGO-counts-{n}.json', 'w') as cp:
        #     json.dump(counts(pred_labels, cluster_test_labels, None, nbins), cp, indent=4)
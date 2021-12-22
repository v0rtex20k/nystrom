import os, json
import numpy as np
import pandas as pd
from typing import *
from small_datasets import *
from algos import *
from sklearn.utils import Bunch
from numpy.random import choice, seed
from timeit import default_timer as timer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import mutual_info_classif

import warnings; warnings.filterwarnings('ignore')

def most_important_pixels(data: np.ndarray, labels: np.ndarray, pkeep: float=0.1, seed:int = None)-> np.ndarray:
    n, mi = data.shape[1], mutual_info_classif(data, labels, random_state=seed)
    mask = pd.Series(mi, np.arange(n)).fillna(0).rank(ascending=False)
    mask = mask.replace(mask.max(),n).to_numpy()
    ranks, cutoff = np.argsort(mask), int(abs(pkeep)*n) if 0 <= abs(pkeep) <= 1 else int(abs(pkeep))
    mask[ranks[cutoff:]], mask[ranks[:cutoff]] = False, True
    return mask.astype(bool).flatten()

if __name__ == "__main__":
    os.system('clear'); 
    print('\tLoading MNIST...')
    mnist_bunch: Bunch = fetch_openml(name='mnist_784', version='1'); nbins = 10; seed(42)
    data, true_labels = [mnist_bunch[c].to_numpy(dtype=int, na_value=0) for c in ['data', 'target']] # MNIST.shape == (70000, 784)
    
    data = data / 255 # normalize
    
    print('\tTraining...')
    train_idxs = choice(data.shape[0], size=data.shape[1]//3, replace=False)
    test_mask = np.ones(data.shape[0], dtype=bool)
    test_mask[train_idxs] = False
    mip_mask = most_important_pixels(data[train_idxs,:], true_labels[train_idxs], pkeep=0.33)

    test_grid = np.ix_(test_mask.flatten(), mip_mask)
    test_data, test_labels = data[test_grid], true_labels[test_mask].flatten()

    print("\t\tStarting 0..."); t0 = timer()
    pred0, score0, size0 = cluster0(test_data, test_labels, nbins, 17, 71)
    print(f"\t\tStarting 1... w/ {s}"); t1 = timer()
    pred1, score1, size1 = cluster1(test_data, test_labels, nbins, 17, 71)
    print("\t\tStarting 3..."); t3 = timer()
    pred3, score3, size3  = cluster3(test_data, 250, 250, test_labels, nbins, 17, 71); fin = timer()
    print(f"ZERO:  {r3(score0)} ({dur(t0,t1)}) " + \
          f"ONE:   {r3(score1)} ({dur(t1,t3)}) " + \
          f"THREE  {r3(score3)} ({dur(t3,fin)}) " + \
          f"RATIO: {r3(score3/score0)}")

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

from multiprocessing import Process
def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

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

    MAX_ITER = 40
    sizes = dict()
    results03 = dict()
    for n in range(1,MAX_ITER+1):
        s: int = n*500
        s_idxs = choice(test_data.shape[0], s, replace=False)
        s_data, s_labels = test_data[s_idxs], test_labels[s_idxs]
        print(f'\tClustering MNIST w/ {n*500}...')

        # print("\t\tStarting 0...");# t0 = timer()
        # pred0, score0, size0 = cluster0(s_data, s_labels, nbins, 17, 71)
        #t1 = timer()
        print("\t\tStarting 3...");# t3 = timer()
        pred3, score3, size3  = cluster3(s_data, 250, 250, s_labels, nbins, 17, 71);# fin = timer()
        print(f"SIZE3: {size3}")
        # sizes[s] = size0
        # results03[s] = { 'a0': (r3(score0), dur(t0,t1)), 'a3': (r3(score3), dur(t3,fin)), 'r': r3(score3/score0)}

    exit()
    # with open('mnist_morning__03__.txt', 'w') as fptr:
    #     json.dump(results03, fptr)
    # with open('mnist_pineapple__0__.txt', 'w') as fptr:
    #     json.dump(sizes, fptr)

    print("\n DUMPED ---> SWITCHING TO ONE  ... \n")
    exit()
    results1 = dict()
    try:
        for n in range(9,15):
            s: int = (n+1)*500
            s_idxs = choice(test_data.shape[0], s, replace=False)
            s_data, s_labels = test_data[s_idxs], test_labels[s_idxs]
            print(f"\t\tStarting 1... w/ {s}"); t1 = timer()
            pred1, score1, size = cluster1(s_data, s_labels, nbins, 17, 71); fin = timer()
            
            results1[s] = {'a1': (r3(score1), dur(t1, fin), size)}
            print(f'\t\t[{s}]: {(r3(score1), dur(t1, fin), size)}')
    except: print("Something exploded...")
    finally:
        with open('mnist_shower__1__.txt', 'w') as fptr:
            json.dump(results1, fptr)
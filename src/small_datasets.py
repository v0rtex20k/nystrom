import os, json
import numpy as np
import pandas as pd
from typing import *
import scipy.io as sio
from algos import *
from datetime import timedelta
from numpy.random import choice, seed
from timeit import default_timer as timer
from sklearn.cluster import MiniBatchKMeans as MBK
import warnings; warnings.filterwarnings('ignore')

r3 = lambda x: round(abs(x),3)
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
                    "labels": MBK(n_clusters=6).fit_predict(pd.read_csv("../data/housing.csv").loc[:, ["Latitude", "Longitude"]].to_numpy()), # labels not known
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

    except FileNotFoundError: exit(f"Could not find \"{name}\" dataset")

dur = lambda a,b: timedelta(seconds=b - a).seconds
if __name__ == "__main__":
        os.system('clear')
        data = load_sample_data("cali", subsample=5000)

        print("\tStarting 0..."); t0 = timer()
        pred0, score0 = cluster0(data["data"], data["labels"], data["n_clusters"], fit_seed=17, score_seed=71)
        print("\tStarting 1..."); t1 = timer()
        pred1, score1, size = cluster1(data["data"], data["labels"], data["n_clusters"], fit_seed=17, score_seed=71)
        print("\tStarting 3..."); t3 = timer()
        pred3, score3 = cluster3(data["data"], data["n_clusters"], data["n_clusters"], data["labels"], data["n_clusters"], fit_seed=17, score_seed=71)
        fin = timer()
        print(f"ZERO:  {r3(score0)} ({dur(t0,t1)}) " + \
              f"ONE:   {r3(score1)} ({dur(t1,t3)}) " + \
              f"THREE  {r3(score3)} ({dur(t3,fin)}) " + \
              f"RATIO: {r3(score3/score0)}")
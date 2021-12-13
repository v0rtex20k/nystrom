import numpy as np
from typing import *
import os, sys, json
from math import floor
import matplotlib.pyplot as mplt
import warnings; warnings.filterwarnings('ignore')

def plot(data: Dict[str, Dict[str, float]], color: str='b')-> None:
    '''
        Plot counts after clustering

        Params
            - data:  Dataset to be plotted
            - color: Bar color, defaults to blue
    '''
    nbins: int = len(data)
    w: int = nbins//2
    (_, a) = mplt.subplots(2, w, sharey=True, tight_layout=True)

    a[0,0].set_ylabel('Label Freqs'); a[1,0].set_ylabel('Label Freqs')
    for i in range(nbins):
        r,c = floor(i/w), i%w
        bin_counts: List[Tuple[int, float]] = [(int(k), round(v*100,2)) for k,v in data[str(i)].items()]
        a[r,c].bar(x=np.arange(nbins), height=[c for _,c in sorted(bin_counts, key=lambda x: x[0])], width=1, color=color, align='center')
        a[r,c].set_xlabel(f'Preds for {i}')
        a[r,c].set_xlim(0,10)
        a[r,c].set_xticks(np.arange(nbins))
    
    mplt.show()


if __name__ == "__main__":
    data: Dict[str, Dict[str, float]] = dict(); count_filepath: str = ""
    try: plots = list(zip(sys.argv[1:], sys.argv[2:]))
    except: exit("You forgot to provide a file or color!")

    for (path, c) in plots:
        if os.path.exists(path):
            with open(path, 'r') as cfp:
                plot(json.load(cfp),c) if 'q' not in input("\tNext? ").strip().lower() else exit("\nBYE")
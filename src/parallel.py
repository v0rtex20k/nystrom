from functools import partial
import numpy as np
from typing import *
from sklearn.utils import Bunch
from nystrom import *
from sklearn.datasets import fetch_openml
from datetime import timedelta

import os

import numpy
_multi=False
_ncpus=1

try:
  # May raise ImportError
  import multiprocessing
  _multi=True

  # May raise NotImplementedError
  _ncpus = multiprocessing.cpu_count()
except:
  pass


__all__ = ('parallel_map',)


def counts(preds: np.ndarray, trues: np.ndarray, idxs: np.ndarray, nbins: int=10)-> float:
    bin_dict = {}
    for i in range(nbins): # true label of 0
        bin_idxs = np.argwhere(trues.flatten()[idxs] == i).flatten()
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


def worker(f, ii, chunk, out_q, err_q, lock):
  """
  A worker function that maps an input function over a
  slice of the input iterable.

  :param f  : callable function that accepts argument from iterable
  :param ii  : process ID
  :param chunk: slice of input iterable
  :param out_q: thread-safe output queue
  :param err_q: thread-safe queue to populate on exception
  :param lock : thread-safe lock to protect a resource
         ( useful in extending parallel_map() )
  """
  vals = []

  # iterate over slice 
  for val in chunk:
    try:
      result = f(val)
    except Exception as e:
      err_q.put(e)
      return

    vals.append(result)

  # output the result and task ID to output queue
  out_q.put( (ii, vals) )


def run_tasks(procs, err_q, out_q, num):
  """
  A function that executes populated processes and processes
  the resultant array. Checks error queue for any exceptions.

  :param procs: list of Process objects
  :param out_q: thread-safe output queue
  :param err_q: thread-safe queue to populate on exception
  :param num : length of resultant array

  """
  # function to terminate processes that are still running.
  die = (lambda vals : [val.terminate() for val in vals
             if val.exitcode is None])

  try:
    for proc in procs:
      proc.start()

    for proc in procs:
      proc.join()

  except Exception as e:
    # kill all slave processes on ctrl-C
    die(procs)
    raise e

  if not err_q.empty():
    # kill all on any exception from any one slave
    die(procs)
    raise err_q.get()

  # Processes finish in arbitrary order. Process IDs double
  # as index in the resultant array.
  results=[None]*num;
  while not out_q.empty():
    idx, result = out_q.get()
    results[idx] = result

  # Remove extra dimension added by array_split
  return list(numpy.concatenate(results))


def parallel_map(function, sequence, numcores=None):
  """
  A parallelized version of the native Python map function that
  utilizes the Python multiprocessing module to divide and 
  conquer sequence.

  parallel_map does not yet support multiple argument sequences.

  :param function: callable function that accepts argument from iterable
  :param sequence: iterable sequence 
  :param numcores: number of cores to use
  """
  if not callable(function):
    raise TypeError("input function '%s' is not callable" %
              repr(function))

  if not numpy.iterable(sequence):
    raise TypeError("input '%s' is not iterable" %
              repr(sequence))

  size = len(sequence)

  if not _multi or size == 1:
    return map(function, sequence)

  if numcores is None:
    numcores = _ncpus

  # Returns a started SyncManager object which can be used for sharing 
  # objects between processes. The returned manager object corresponds
  # to a spawned child process and has methods which will create shared
  # objects and return corresponding proxies.
  manager = multiprocessing.Manager()

  # Create FIFO queue and lock shared objects and return proxies to them.
  # The managers handles a server process that manages shared objects that
  # each slave process has access to. Bottom line -- thread-safe.
  out_q = manager.Queue()
  err_q = manager.Queue()
  lock = manager.Lock()

  # if sequence is less than numcores, only use len sequence number of 
  # processes
  if size < numcores:
    numcores = size 

  # group sequence into numcores-worth of chunks
  sequence = numpy.array_split(sequence, numcores)

  procs = [multiprocessing.Process(target=worker,
           args=(function, ii, chunk, out_q, err_q, lock))
         for ii, chunk in enumerate(sequence)]

  return run_tasks(procs, err_q, out_q, numcores)


from timeit import default_timer as timer
dur = lambda a,b: timedelta(seconds=b-a)
if __name__ == "__main__":
    print('\tLoading MNIST...')
    mnist_bunch: Bunch = fetch_openml(name='mnist_784', version='1')
    data, true_labels = [mnist_bunch[c].to_numpy(dtype=int, na_value=0) for c in ['data', 'target']]
    data = data/255 # data.shape == (70000, 784)
    
    print("PARTIAL")
    nbins = 10
    ng = partial(ng_nystrom, k=nbins, gamma=None, seed=17)
    print("SPLITTING")
    iterable = np.array_split(data, 10)
    print("MAPPING")
    start = timer()
    parallel_pred_labels = list(parallel_map(ng, iterable , numcores=os.cpu_count()))
    stop  = timer()
    print(f"parallel map in {dur(start, stop)} secs")
    import json
    print('\tAssessing performance and saving results...')
    for i, ppl in enumerate(parallel_pred_labels):
        with open(f'parallel-counts-{i}.json', 'w') as cp:
            json.dump(counts(ppl, true_labels, np.arange(ppl.shape[0]), nbins), cp, indent=4)
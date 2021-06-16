import numpy as np
import time
import argparse
import os
import logging

from mpi4py import MPI
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('rank {} is starting'.format(rank))

if rank == 0:
    w = csr_matrix(np.zeros(10))
else:
    w = csr_matrix(np.random.uniform(size=10))
w = comm.reduce(w)
if rank == 0:
    print(w)
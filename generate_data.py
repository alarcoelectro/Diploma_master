import numpy as np
import os
import argparse
import random

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from scipy.sparse import csr_matrix

import config

parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')
parser.add_argument('--n_workers', action='store', dest='n_workers', type=int, default=None,
                    help='Number of workers that will be used')
parser.add_argument('--dataset', action='store', dest='dataset', default=None, help='The name of the dataset')

args = parser.parse_args()
n_workers = args.n_workers
dataset = config.dataset
use_local_data = config.use_local_data

if n_workers is None:
    n_workers = config.n_workers

if dataset is None:
    dataset = config.dataset
    
def load_data(dataset):
    # data = load_svmlight_file("{0}/{1}".format(config.datasets_path, dataset), zero_based=zero_based.get(dataset, 'auto'))
    data = load_svmlight_file("{0}/{1}".format(config. datasets_path, dataset), zero_based='auto')
    return data[0], data[1]

Xs = []
ys = []

# Remove old data
os.system("bash -c 'rm {0}/data/*'".format(config.scripts_path))

# Load new data
X, y = load_data(dataset)
if 2 in y:
    y[y == 2] = -1
y = csr_matrix(y).T
if config.permute_data:
    perm = np.arange(X.shape[0])
    random.shuffle(perm)
    X = X[perm]
    y = y[perm]

data_len, d = X.shape
print('Number of data points:', data_len)
sep_idx = [0] + [(data_len * i) // n_workers for i in range(1, n_workers)] + [data_len]
L = 0.25 * np.max(X.multiply(X).sum(axis=1))
l1_penalties = config.l1_penalties
# l1 = l1_penalties[dataset]
l1 = 0 # We are not interested in prox right now
data_info = [data_len, d, l1, L]

# Save data for workers
workers_data_path = "{}/data".format(config.scripts_path)
if not os.path.exists(workers_data_path):
    os.makedirs(workers_data_path)

if use_local_data:
    for i in range(n_workers):
        print('Creating chunk number', i + 1)
        start, end = sep_idx[i], sep_idx[i + 1]
        L_i = 0.25 * np.max(X[start:end].multiply(X[start:end]).sum(axis=1))
        data_info.append(L_i)
        dump_svmlight_file(X[start:end], y[start:end], "{0}/data/{1}".format(config.scripts_path, i), zero_based=True)
else:
    data_info += [L] * n_workers
    print('All workers use full dataset and no extra data files were created.')
    
np.save('{}/data/data_info'.format(config.scripts_path), data_info)
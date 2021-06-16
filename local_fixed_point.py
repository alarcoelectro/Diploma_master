import numpy as np
import time
import argparse
import os
import logging
import sys
import scipy
from numpy.random import geometric

from mpi4py import MPI
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

import config
from config import big_regularization, dataset, logs_path, use_local_data
from functions import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size()

    
logging.basicConfig(filename='local_fixed_point_out.log', level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__))
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception
try:

    parser = argparse.ArgumentParser(description='Run local fixed-point method')
    parser.add_argument('--it', action='store', dest='it_max', type=int, help='Maximum number of iterations')
    parser.add_argument('--t', action='store', dest='t_max', type=float, help='Time limit')
    parser.add_argument('--lr0', action='store', dest='lr0', type=float, default=1,
                        help='Learning rate relative to smoothness, defines fixed point operator')
    parser.add_argument('--operator', action='store', dest='operator', type=str, default="full",
                        help='Fixed point operator, options: full, sequential')
    parser.add_argument('--alpha', action='store', dest='alpha', type=float, default=0.5, 
                        help='Learning rate of Algorithm 1')
    parser.add_argument('-loopless', action='store_true', dest='loopless',
                        help='Run loopless version of algorithm')
    parser.add_argument('--prob',action='store',dest='comm_prob', type=float, default=0.5,
                        help='Probability of communication for loopless varian of the algorithm')
    parser.add_argument('--n_loc', action='store', dest='local_steps', type=int, default=1,
                        help='Number of local steps to perform by each worker')
    parser.add_argument('--out', action='store', dest='output_size', type=int, default=500,
                        help='Number of checkpoints to save')


    args = parser.parse_args()
    it_max = args.it_max
    t_max = args.t_max
    lr0 = args.lr0
    operator = args.operator
    alpha = args.alpha
    loopless = args.loopless
    comm_prob = args.comm_prob
    local_steps = args.local_steps
    output_size = args.output_size

    
    data_path = '{}/data'.format(config.scripts_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    dataset_full_path = '{0}/{1}'.format(config.datasets_path, dataset)

    
    if it_max is None or it_max == 0:
        it_max = np.inf
    if t_max is None or t_max == 0:
        t_max = np.inf
    if (it_max is np.inf) and (t_max is np.inf):
        if rank == 0:
            print('At least one stopping criterion must be specified')
        
        raise ValueError()

       

    data_info = np.load('{}/data_info.npy'.format(data_path))
    N = int(data_info[0])
    d = int(data_info[1])
    l1 = data_info[2]
    L = data_info[3]
    Ls = data_info[4:]
    l2 = L / N if big_regularization else 0.1 * L / N
    L_max = np.max(Ls)
    
    lr = lr0 / L


    experiment = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}".format(
        alpha,
        local_steps,
        n_workers,
        dataset,
        operator,
        'big' if big_regularization else 'small',
        'l' if use_local_data else 'f',
        'loopless' if loopless else 'not_loopless',
        comm_prob
#        , time.strftime("%d-%b-%Y-%I:%M:%S")
    )
    
        
    if use_local_data:
        X, y = load_svmlight_file("{0}/{1}".format(data_path, rank), zero_based='auto', n_features=d)
    else:
        X, y = load_svmlight_file(dataset_full_path, zero_based='auto', n_features=d)
    y = csr_matrix(y).T
    if 2 in y:
        y[y == 2] = -1
    n = X.shape[0]
    w = csr_matrix(np.zeros(shape=d)).T
    it = 0
    time_computing = 0
    time_communicating = 0
    max_grad_norm = 0
    
    def fixed_point_full_batch_step(w, X, y, lr, l2):
        w_new = w - lr * logistic_stochastic_grad(w=w, X=X, y=y, l2=l2, batch_size=None)
        return w_new    
    
    def fixed_point_sequential_step(w, X, y, lr, l2):
        w_new = w.copy()
        for ind in range(X.shape[0]):
            w_new = w_new - lr * logistic_stochastic_grad(w=w_new, X=X[ind], y=y[ind], l2=l2, batch_size=None)
        return w_new 
        

    if rank == 0:
        ws = [w.copy()]
        information_sent = [0]
        ts = [0]
        its = [0]
        t_start = time.time()
        information = 0
        t = 0
        it = 0
        max_progress = 0
        logger.debug("Initial stepsize is {}".format(lr0 / L))


    stop = False
    while not stop:
        time_checkpoint = time.time()
        if loopless:
            if rank == 0:
                local_steps = geometric(p=comm_prob, size=1)[0]
            comm.bcast(local_steps, root=0)
        for local_step in range(local_steps):
            if operator == "full":
                T_w = fixed_point_full_batch_step(w=w, X=X, y=y, lr=lr, l2=l2)
                w += alpha * (T_w - w)
            
            if operator == "sequential":
                T_w = fixed_point_sequential_step(w=w, X=X, y=y, lr=lr, l2=l2)
                w += alpha * (T_w - w)
        time_computing += time.time() - time_checkpoint

        time_checkpoint = time.time()
        w = comm.allreduce(w) / n_workers
        if rank == 0:
            stop = time.time() - t_start >= t_max or it >= it_max
        stop = comm.bcast(stop, root=0)
        time_communicating += time.time() - time_checkpoint

        it += 1
        if rank == 0:
            information += d # No compression
            t = time.time() - t_start
            time_progress = int((output_size - 10) * t / t_max)
            iterations_progress = int((output_size - 10) * (it / it_max))
            if (max(time_progress, iterations_progress) > max_progress) or (it <= 10) or stop:
                ws.append(w.copy())
                ts.append(t)
                its.append(it)
                information_sent.append(information)
            max_progress = max(time_progress, iterations_progress)
#     max_grad_norms = np.empty(n_workers)
#     comm.Gather(max_grad_norm, max_grad_norms)
#     max_grad_norm = np.max(max_grad_norms)

    if rank == 0:
        logger.debug("Stepsize after {} iterations is {}".format(it, lr))
        logger.debug('Maximum encountered gradient norm was {}'.format(max_grad_norm))

    if (rank == 1) and config.time_tracking:
        logger.debug('Rank 1 spent {:.3f}%% of time in communication'.format(
            100 * time_communicating / (time_communicating + time_computing)))

    if rank == 0:
        X, y = load_svmlight_file(dataset_full_path, zero_based='auto', n_features=d)
        if 2 in y:
            y[y == 2] = -1
        y = csr_matrix(y).T
        N_X = X.shape[0]
        if N_X != N:
            raise ValueError('The dataset provided in the path has a different \
                              length than in the generated data info')

        print("Saving {0} checkpoints out of {1} total iterates".format(len(ws), its[-1]))
        loss = np.array([reg_logistic_loss(w, X, y, l1, l2) for w in ws])
        np.save("{0}/loss_{1}".format(logs_path, experiment), np.array(loss))
        np.save("{0}/time_{1}".format(logs_path, experiment), np.array(ts))
        np.save("{0}/information_{1}".format(logs_path, experiment), np.array(information_sent))
        np.save("{0}/iteration_{1}".format(logs_path, experiment), np.array(its))
        np.save("{0}/iterates_{1}".format(logs_path, experiment), np.array(ws))
        print('Loss:', loss[::10])
        print('Last iterate density: {:.4f}'.format(ws[-1].count_nonzero() / d))

        if config.plot_convergence:
            import matplotlib.pyplot as plt
            plt.plot(its[:-10], loss[:-10] - np.min(loss))
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.show()

    print("Rank %d is down" % rank)
except Exception as e:
    logging.exception(e)

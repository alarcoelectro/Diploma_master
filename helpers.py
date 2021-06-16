import numpy as np
import os
import itertools
import subprocess

from algorithms import svrg


def get_solution(X, y, L, data_path, l2=None, dataset='mushrooms', big_regularization=True, S=None, lr=None, recompute=False):
    if S is None:
        S = 100 if big_regularization else 1000
    if lr is None:
        lr = 0.15 / L
    n, d = X.shape
    regularization = 'big' if big_regularization else 'small'
    data_info = np.load('{}/data_info.npy'.format(data_path))
    N, L = data_info[:2]
    Ls = data_info[2:]
    if l2 is None:
        l2 = L / N * (1 if big_regularization else 1e-1)
    solution_path = './solutions/w_star_{0}_{1}.npy'.format(dataset, regularization)
    if not os.path.exists('./solutions'):
        os.makedirs('./solutions')
    
    if os.path.isfile(solution_path) and not recompute:
        print('Loading the solution from file')
        print(solution_path)
        w_star = np.load(solution_path)
        ws_svrg = None
    else:
        print('Computing the solution using SVRG')
        ws_svrg, _, _ = svrg(np.zeros(d), X, y, lr=lr, mu=l2, S=S, p=None, t_max=np.inf)
        w_star = ws_svrg[-1]
        np.save(solution_path, w_star)
    return w_star, ws_svrg


def run_local_fixed_point_method_single_experiment(n_workers, scripts_path, it_max, t_max, lr0, operator, alpha, n_local, loopless=False, comm_prob=0.5): 
    code_file_name = 'local_fixed_point.py'
    process = subprocess.Popen("mpiexec -n {0} python {1}/{2} --it {3} --t {4} \
               --lr0 {5} --operator {6} --alpha {7} --n_loc {8} {9}".format(
        n_workers,
        scripts_path,
        code_file_name,
        it_max,
        t_max,
        lr0,
        operator,
        alpha,
        n_local,
        '' if not loopless else '-loopless --prob {}'.format(comm_prob)
    ).split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if len(error) != 0:
        print(error)
    print('#', end='')
    
    
def run_local_fixed_point_method(n_workers, scripts_path, it_max, t_max, lr0s, operators, alphas, local_steps, all_combinations=False):
    if it_max is np.inf or it_max is None:
        raise ValueError('Argparse does not take None or np.inf as an argument. Set it to 0 to put no limit.')
    if t_max is np.inf or t_max is None:
        raise ValueError('Argparse does not take None or np.inf as an argument. Set t to 0 to put no limit.')
                
    code_file_name = 'local_fixed_point.py'
    if all_combinations:
        for params in itertools.product(lr0s, operators, alphas, local_steps):
            lr0, operator, alpha, n_local = params
#            run_local_fixed_point_method_single_experiment(n_workers, scripts_path, code_file_name, it_max, t_max, lr0, operator, alpha, n_local)
            run_local_fixed_point_method_single_experiment(n_workers, scripts_path, code_file_name, it_max, t_max, *params)
    else:
        for batch, lr0, lr_decay, n_local in zip(batches, lr0s, lr_decays, local_steps):
#            run_local_fixed_point_method_single_experiment(n_workers, scripts_path, code_file_name, it_max, t_max, lr0, operator, alpha, n_local)
            run_local_fixed_point_method_single_experiment(n_workers, scripts_path, code_file_name, it_max, t_max, *params)


def run_local_sgd(n_workers, scripts_path, it_max, t_max, batches, lr0s, lr_decays, local_steps, all_combinations=False, use_np=False):
    if it_max is np.inf or it_max is None:
        raise ValueError('Argparse does not take None or np.inf as an argument. Set it to 0 to put no limit.')
    if t_max is np.inf or t_max is None:
        raise ValueError('Argparse does not take None or np.inf as an argument. Set t to 0 to put no limit.')
    if np.inf in batches or None in batches:
        raise ValueError('Argparse does not take None or np.inf as an argument. Set batch to 0 to use full gradient.')
    if all_combinations:
        for params in itertools.product(batches, lr0s, lr_decays, local_steps):
            batch, lr0, lr_decay, n_local = params
            code_file_name = 'local_sgd_np.py' if use_np else 'local_sgd.py'
            process = subprocess.Popen("mpiexec -n {0} python {1}/{2} --it {3} --t {4} --batch {5} \
                       --lr0 {6} --lr_t {7} --n_loc {8}".format(
                n_workers,
                scripts_path,
                code_file_name,
                it_max,
                t_max,
                batch,
                lr0,
                lr_decay,
                n_local,
            ).split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            if len(error) != 0:
                print(error)
            print('#', end='')
    else:
        for batch, lr0, lr_decay, n_local in zip(batches, lr0s, lr_decays, local_steps):
            process = subprocess.Popen("mpiexec -n {0} python {1}/local_sgd.py --it {2} --t {3} --batch {4} \
                       --lr0 {5} --lr_t {6} --n_loc {7}".format(
                n_workers,
                scripts_path,
                it_max,
                t_max,
                batch,
                lr0,
                lr_decay,
                n_local,
            ).split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            if len(error) != 0:
                print(error)   
            print('#', end='')


def run_local_diana(n_workers, scripts_path, it_max, t_max, batches, lr0s, lr_decays, synch_probs, all_combinations=False):
    if it_max is np.inf or it_max is None:
        raise ValueError('Argparse does not take None or np.inf as an argument. Set it to 0 to put no limit.')
    if t_max is np.inf or t_max is None:
        raise ValueError('Argparse does not take None or np.inf as an argument. Set t to 0 to put no limit.')
    if np.inf in batches or None in batches:
        raise ValueError('Argparse does not take None or np.inf as an argument. Set batch to 0 to use full gradient.')
    for params in itertools.product(batches, lr0s, lr_decays, synch_probs):
        batch, lr0, lr_decay, synch_prob = params
        process = subprocess.Popen("mpiexec -n {0} python {1}/local_diana.py --it {2} --t {3} --batch {4} \
                   --lr0 {5} --lr_t {6} --pr {7}".format(
            n_workers,
            scripts_path,
            it_max,
            t_max,
            batch,
            lr0,
            lr_decay,
            synch_prob,
        ).split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        if len(error) != 0:
            print(error)
        print('#', end='')
        
def read_logs_local_fixed_point(alpha, local_steps, n_workers, dataset, operator, big_regularization, use_local_data, logs_path='/', loopless=False, prob=0.5):
    input_params = (
        alpha,
        local_steps,
        n_workers,
        dataset,
        operator,
        'big' if big_regularization else 'small',
        'l' if use_local_data else 'f',
        'loopless' if loopless else 'not_loopless',
        prob
    )

    format_suffix = '_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}'
    ws = np.load('{}/iterates{}.npy'.format(logs_path, format_suffix).format(*input_params), allow_pickle=True)
    loss = np.load('{}/loss{}.npy'.format(logs_path, format_suffix).format(*input_params), allow_pickle=True)
    its = np.load('{}/iteration{}.npy'.format(logs_path, format_suffix).format(*input_params), allow_pickle=True)
    ts = np.load('{}/time{}.npy'.format(logs_path, format_suffix).format(*input_params), allow_pickle=True)
    return ws, loss, its, ts
    
        
def read_logs_local_sgd(batch, lr0, lr_decay, local_steps, big_regularization, use_local_data, logs_path='/'):
    # Cast int to float
    lr0 *= 1.
    lr_decay *= 1.
    input_params = (
        batch,
        lr0,
        lr_decay,
        local_steps,
        'big' if big_regularization else 'small',
        'l' if use_local_data else 'f'
    )

    format_suffix = '_{0}_{1}_{2}_{3}_{4}_{5}'
    ws = np.load('{}/iterates{}.npy'.format(logs_path, format_suffix).format(*input_params), allow_pickle=True)
    loss = np.load('{}/loss{}.npy'.format(logs_path, format_suffix).format(*input_params), allow_pickle=True)
    its = np.load('{}/iteration{}.npy'.format(logs_path, format_suffix).format(*input_params), allow_pickle=True)
    ts = np.load('{}/time{}.npy'.format(logs_path, format_suffix).format(*input_params), allow_pickle=True)
    return ws, loss, its, ts


def read_logs_diana(batch, lr0, lr_decay, synch_prob, big_regularization, use_local_data, logs_path='/'):
    # Cast int to float
    lr0 *= 1.
    lr_decay *= 1.
    synch_prob *= 1.
    input_params = (
        batch,
        lr0,
        lr_decay,
        synch_prob,
        'big' if big_regularization else 'small',
        'l' if use_local_data else 'f'
    )

    format_suffix = '_{0}_{1}_{2}_{3}_{4}_{5}'
    ws = np.load('{}/iterates{}.npy'.format(logs_path, format_suffix).format(*input_params))
    loss = np.load('{}/loss{}.npy'.format(logs_path, format_suffix).format(*input_params))
    its = np.load('{}/iteration{}.npy'.format(logs_path, format_suffix).format(*input_params))
    ts = np.load('{}/time{}.npy'.format(logs_path, format_suffix).format(*input_params))
    return ws, loss, its, ts
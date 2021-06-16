import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from np_functions import *

def svrg(x_init, X, y, lr, mu=0, S=50, p=None, t_max=np.inf):
    n, d = X.shape
    assert(len(x_init) == d)
    assert(len(y) == n)
    if p is None:
        p = 1 / n
    x = np.array(x_init)
    xs = []
    its = []
    ts = []
    t_start = time.time()
    step = min(int(S / p) // 200, int(1 / (2 * p)))
    
    s = 0
    it = 0
    while s < S and time.time() - t_start < t_max:
        full_grad = gradient(x, X, y, mu)
        x0 = np.copy(x)
        continue_loop = True
        while continue_loop and (time.time() - t_start <= t_max):
            i = random.randrange(n)
            grad = logreg_sgrad(x, X[i], y[i], mu)
            old_grad = logreg_sgrad(x0, X[i], y[i], mu)
            v = grad - old_grad + full_grad
            x -= lr * v
            if it % step == 0 or (it <= 10):
                xs.append(np.copy(x))
                its.append(it)
                ts.append(time.time() - t_start)
            it += 1
            if np.random.uniform(0, 1) < p:
                continue_loop = False
        s += 1
        print('#', end='')
    return np.array(xs), np.array(its), np.array(ts)
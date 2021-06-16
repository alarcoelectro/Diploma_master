import numpy as np
import random
import time
import scipy

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.utils.extmath import safe_sparse_dot

supported_penalties = ['l1']


def safe_sparse_add(a, b):
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        # both are sparse, keep the result sparse
        return a + b
    else:
        # on of them is non-sparse, convert
        # everything to dense.
        if scipy.sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b.ravel()
        elif scipy.sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b

def logreg_loss(x, A, b, l2):
    assert l2 >= 0
    assert len(b) == A.shape[0]
    assert len(x) == A.shape[1]
    l = np.log(1 + np.exp(-A.dot(x) * b))
    m = b.shape[0]
    return np.sum(l) / m + l2/2 * norm(x) ** 2

def logreg_grad(w, X, y, mu):
    assert mu >= 0
    assert (len(y) == X.shape[0])
    assert (len(w) == X.shape[1])
    loss_grad = np.mean([logreg_sgrad(w, X[i], y[i]) for i in range(len(y))], axis=0)
    assert len(loss_grad) == len(w)
    return loss_grad + mu * w

def logreg_sgrad(w, x_i, y_i, mu=0):
    assert mu >= 0
    assert len(w) == len(x_i)
    assert y_i in [-1, 1]
    loss_sgrad = - y_i * x_i / (1 + np.exp(y_i * np.dot(x_i, w)))
    assert len(loss_sgrad) == len(w)
    return loss_sgrad + mu * w

def sample_logreg_sgrad(w, X, y, mu=0, batch=1):
    assert mu >= 0
    n, d = X.shape
    assert(len(w) == d)
    assert(len(y) == n)
    grad_sum = 0
    for b in range(batch):
        i = random.randrange(n)
        grad_sum += logreg_sgrad(w, X[i], y[i], mu)
    return grad_sum / batch

def r(x, l1):
    return l1 * norm(x, ord = 1)

def F(x, A, b, l2, l1=0):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert ((l2 >= 0) & (l1 >= 0))
    return logreg_loss(x, A, b, l2) + r(x, l1)

def prox_r(x, gamma, coef, penalty='l1'):
    assert penalty in supported_penalties
    assert(gamma > 0 and coef >= 0)
    if penalty == 'l1':
        l1 = coef
        return x - abs(x).minimum(l1 * gamma).multiply(x.sign())
    
def gradient(w, X, y_, l2, normalize=True):
    y = (y_ + 1) / 2 if -1 in y_ else y_
    activation = scipy.special.expit(safe_sparse_dot(X, w, dense_output=True).ravel())
    grad = safe_sparse_add(X.T.dot(activation - y) / X.shape[0], l2 * w)
    grad = np.asarray(grad).ravel()
    if normalize:
        return grad
    return grad * len(y)
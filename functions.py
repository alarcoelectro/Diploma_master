import random
import numpy as np

from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix

def logistic_full_grad(w, X, y, l2):
    assert ((y.shape[0] == X.shape[0]) & (w.shape[0] == X.shape[1]))
    assert l2 >= 0
    denom = csr_matrix(1 / (1 + np.exp(X.dot(w).multiply(y).A)))
    g = -(X.multiply(y).multiply(denom).mean(axis=0).T)
    return csr_matrix(g) + l2 * w
    
def logistic_stochastic_grad(w, X, y, l2, batch_size=None):
    n, d = X.shape
    if batch_size is None:
        return logistic_full_grad(w, X, y, l2)
    idx = random.sample(range(n), k=batch_size)
    return logistic_full_grad(w, X[idx], y[idx], l2)
    
def prox(w, lr, coef, penalty='l1'):
    assert (lr > 0) and (coef >= 0)
    w_shape = w.shape
    if penalty == 'l1':
        l1 = coef
        w_sign = w.sign()
        w_thresholded = (abs(w) - lr * l1 * abs(w_sign)).maximum(0)
        w = w_thresholded.multiply(w_sign)
        w.eliminate_zeros()
        assert(w.shape == w_shape)
        return w
    
def unreg_logistic_loss(w, X, y):
    l = np.log(1 + np.exp(-X.dot(w).multiply(y).A))
    return np.mean(l)
        
def reg_logistic_loss(w, X, y, l1, l2):
    assert (y.shape[0] == X.shape[0]) and (w.shape[0] == X.shape[1])
    assert (l2 >= 0) and (l1 >= 0)
    return unreg_logistic_loss(w, X, y) + l1 * norm(w, ord=1) + l2 / 2 * norm(w) ** 2
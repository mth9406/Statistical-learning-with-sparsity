import numpy as np
from .utils import soft_thresholding_fn, optimal_check_by_tol


class MyLasso(object):

    # hyper parameter setttings.
    def __init__(self, 
                 l1_penalty=0.7,
                 normalize= True
                 ):
        self.l1_penalty = l1_penalty
        self.normalize = normalize

    def fit(self, X, y, l1_penalty= None, initial_beta = None, max_iter = 1000, verbose = True):
        mean, std = np.mean(X, axis = 1, keepdims=True), np.std(X, axis=1, keepdims=True)
        mean_y = np.mean(y, axis=-1, keepdims=True)
        if l1_penalty is None:
            l1_penalty = self.l1_penalty
    
        self.mean, self.std, self.mean_y = mean, std, mean_y
        n, p = X.shape
        Z = (X-mean)/std # normalize
        y_c = y - mean_y # centered

        # coefficients to optimize
        if initial_beta is None:
            self.beta = np.zeros(shape= (p,1))
        else:
            assert initial_beta.shape == (p,1)
            self.beta = initial_beta
        idx = np.arange(p)
        if verbose:
            print('Start searching for the optimal beta...')
        num_iter = 0
        is_optimal = False
        while (not is_optimal) and num_iter < max_iter:
            cor_seq = np.random.permutation(p) # updates jth beta in an arbitrary order
            beta_old = np.copy(self.beta)
            for j in cor_seq:
                idx_j = (idx != j)
                r_j = y_c - Z[:,idx_j]@self.beta[idx_j,0] # residual n,1 
                lse = (Z[:,j].T@r_j)/n # 1,n  n,1
                self.beta[j,0] = soft_thresholding_fn(lse, l1_penalty)
            num_iter += 1
            # is_optimal = optimal_check_by_subgradient(Z, y_c, self.beta, l1_penalty)
            is_optimal = optimal_check_by_tol(beta_old, self.beta)
        if verbose:
            res = 'Optimal beta found.' if is_optimal else 'Convergence failed.' 
            print(res)

    def predict(self, X):
        Z = (X-self.mean)/self.std
        return Z@self.beta + self.mean_y   

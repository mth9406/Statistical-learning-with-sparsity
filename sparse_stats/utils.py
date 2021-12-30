import numpy as np

def soft_thresholding_fn(x, l1_penalty):
    return np.sign(x)*np.maximum(np.abs(x)-l1_penalty,0)


# depreciated.
def lasso_objective_fn(y, y_pred, n, l1_penalty, beta):
    n = X.shape[0]
    return (np.linalg.norm(y-y_pred, ord=2)**2)/(2*n) + l1_penalty * np.linalg.norm(beta, ord=1)

# depreciated.
def optimal_check_by_subgradient(X, y, beta, l1_penalty, eps = 1e-6):
    n, p = X.shape
    is_optimal = True
    # beta = beta.reshape((-1,1))
    for j in range(p):
        sub_grad_cond = -(X[:,j].T@(y[:,np.newaxis]-X@beta))/n + l1_penalty*np.sign(beta[j,0])
        if sub_grad_cond >= eps:
            is_optimal = False
            return is_optimal
    return is_optimal

def optimal_check_by_tol(beta_old, beta_new, tol= 1e-6):
    is_optimal = np.linalg.norm(beta_old-beta_new) < tol
    return is_optimal

def find_max_l1_penalty(X,y):
    max_l1_penalty = -1
    n, p = X.shape
    Z = (X - np.mean(X, axis=1, keepdims= True))/np.std(X, axis=1, keepdims= True)
    y_c = y - np.mean(y, keepdims= True)
    for j in range(p):
        tmp_l1_penalty = np.abs(Z[:,j].T@y_c[:,np.newaxis])/n
        if tmp_l1_penalty > max_l1_penalty:
            max_l1_penalty = tmp_l1_penalty
    return max_l1_penalty

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)


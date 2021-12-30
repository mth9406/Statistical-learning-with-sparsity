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


def l1_penalty_tunning_auto(X_tr, y_tr, X_te, y_te, n_l1_penalties = 50, n_l1_penaly_minimum= 0 ,lasso_regr= None):
    
    if lasso_regr is None:
        lasso_regr = MyLasso()

    n, p = X_tr.shape
    # to save beta
    betas = np.zeros(shape = (p, n_l1_penalties))
    # max l1 penalty
    max_l1_penalty = find_max_l1_penalty(X_tr, y_tr)
    # candidates for l1_penalties (log_scaled)
    l1_penalties = np.logspace(max_l1_penalty, n_l1_penaly_minimum, num= n_l1_penalties)
    # scores
    scores = np.zeros(shape = n_l1_penalties)

    print('Start finding the optimal beta... along the n_l1_penalties')

    print('iter 1 started')
    lasso_regr.fit(X_tr, y_te, l1_penalty=max_l1_penalty, verbose = True)
    betas[:,0] = lasso_regr.beta.flatten()
    scores[0] = mse(y_te, lasso_regr.predict(X_te))
    print('iter 1 done.')

    for i in np.arange(1, n_l1_penalties):
        print(f'iter {i+1} started')
        lasso_regr.fit(X_tr, y_tr, l1_penalty=l1_penalties[i],initial_beta= betas[:,i-1:i], verbose = True)
        betas[:,i] = lasso_regr.beta.flatten()
        scores[i] = mse(y_te, lasso_regr.predict(X_te))
        print(f'iter {i+1} done.')

    best_param_idx = np.argmin(scores)
    best_beta = betas[:, best_param_idx:best_param_idx+1]

    lasso_regr.beta = best_beta

    return betas, scores, l1_penalties, best_beta, lasso_regr


def l1_penalty_tunning(X_tr, y_tr, X_te, y_te, l1_penalties, lasso_regr= None):
    
    if lasso_regr is None:
        lasso_regr = MyLasso()

    n, p = X_tr.shape
    n_l1_penalties = len(l1_penalties)
    # to save beta
    betas = np.zeros(shape = (p, n_l1_penalties))

    # scores
    scores = np.zeros(shape = n_l1_penalties)

    print('Start finding the optimal beta... along the n_l1_penalties')

    print('iter 1 started')
    lasso_regr.fit(X_tr, y_te, l1_penalty=l1_penalties[0], verbose = True)
    betas[:,0] = lasso_regr.beta.flatten()
    scores[0] = mse(y_te, lasso_regr.predict(X_te))
    print('iter 1 done.')

    for i in np.arange(1, n_l1_penalties):
        print(f'iter {i+1} started')
        lasso_regr.fit(X_tr, y_tr, l1_penalty=l1_penalties[i],initial_beta= betas[:,i-1:i], verbose = True)
        betas[:,i] = lasso_regr.beta.flatten()
        scores[i] = mse(y_te, lasso_regr.predict(X_te))
        print(f'iter {i+1} done.')

    best_param_idx = np.argmin(scores)
    best_beta = betas[:, best_param_idx:best_param_idx+1]

    lasso_regr.beta = best_beta

    return betas, scores, l1_penalties, best_beta, lasso_regr


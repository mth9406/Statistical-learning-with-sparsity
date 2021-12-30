import numpy as np
from .Lasso import MyLasso

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

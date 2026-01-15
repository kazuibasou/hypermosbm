import time
import numpy as np
import math
from scipy.sparse import csr_matrix
import hypergraph
from scipy.special import loggamma
from hypergraphx.communities.hy_mmsbm.model import HyMMSBM
from hypergraphx.core.hypergraph import Hypergraph
import random

def fit_hymmsbm(H: hypergraph.HyperGraph, K: int, random_state=None,
                n_init=10, max_iter=500, tol=None, check_convergence_every=1):

    np.random.seed(random_state)
    random.seed(random_state)

    H_ = Hypergraph(H.E, weighted=True, weights=H.A)
    for v in set(range(0, H.N)) - set(H_.get_nodes()):
        H_.add_node(v)

    best_model = None
    best_loglik = float("-inf")
    for j in range(n_init):

        model = HyMMSBM(
            K=K,
            assortative=False
        )
        model.fit(
            H_,
            n_iter=max_iter,
            tolerance=tol,
            check_convergence_every=check_convergence_every
        )

        log_lik = model.log_likelihood(H_)
        if log_lik > best_loglik:
            best_model = model
            best_loglik = log_lik

    best_param = (best_model.u, best_model.w)

    return best_loglik, best_param

def calc_log_binom(n, k):
    if k < 0 or k > n:
        return -np.inf
    return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

def calc_hyperedge_score(hyperedge: tuple, U: np.ndarray, W: np.ndarray, EPS=1e-20) -> float:

    if not hyperedge or U is None or W is None:
        return 0.0

    nodes_indices = list(hyperedge)
    U_e = U[nodes_indices, :]

    S_e = U_e.sum(axis=0)
    first_term = S_e @ W @ S_e.T
    second_term = np.sum((U_e @ W) * U_e)

    lambda_e = 0.5 * (first_term - second_term)
    if lambda_e <= 0.0:
        lambda_e = EPS
    log_lambda_e = np.log(lambda_e)

    n = int(U.shape[0])
    s = len(hyperedge)
    log_kappa_e = math.log(float(s * (s - 1)) / 2.0) + calc_log_binom(n - 2, s - 2)

    return log_lambda_e - log_kappa_e

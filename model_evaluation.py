import hypergraph
import numpy as np
import random
import time
import hymmsbm, hypermosbm
#import hypergraphmt
#from hypergraphx.communities.hypergraph_mt.model import HypergraphMT
#from hypergraphx.core.hypergraph import Hypergraph
#import warnings
import math
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

def construct_train_and_test_sets(H: hypergraph.HyperGraph, n_splits: int,
                                  num_samples=10000, seed=None, model_evaluation=True):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_indices = np.arange(H.M)

    train_and_test_sets = []

    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(all_indices)):

        E_train = [H.E[idx] for idx in train_indices]
        A_train = [H.A[idx] for idx in train_indices]
        H_train = hypergraph.HyperGraph(H.N, len(E_train))
        H_train.E = E_train
        H_train.A = A_train

        E_test = [H.E[idx] for idx in test_indices]
        A_test = [H.A[idx] for idx in test_indices]
        H_test = hypergraph.HyperGraph(H.N, len(E_test))
        H_test.E = E_test
        H_test.A = A_test

        original_edges_set = set(tuple(sorted(e)) for e in H.E)

        if model_evaluation:
            current_seed = seed + fold_idx if seed is not None else None
            rng = random.Random(current_seed)

            E_lp_pos_pool = E_test

            if not E_lp_pos_pool:
                E_lp_pos = []
            else:
                E_lp_pos = rng.choices(E_lp_pos_pool, k=num_samples)

            E_lp_neg = []
            for e_pos in E_lp_pos:
                s = len(e_pos)
                while True:
                    e_neg = tuple(sorted(rng.sample(range(H.N), k=s)))
                    if e_neg not in original_edges_set:
                        E_lp_neg.append(e_neg)
                        break
        else:
            E_lp_pos, E_lp_neg = [], []

        train_and_test_sets.append(
            (
                H_train,
                H_test,
                (E_lp_pos, E_lp_neg),
            )
        )

    return train_and_test_sets

class ModelEvaluator():

    H = None
    K = None
    hs_clusters = None
    train_and_test_sets = []
    model_name = None
    EPS = 1e-20

    def __init__(self, H: hypergraph.HyperGraph, model_name: str, K: int, hs_clusters: list or None,
                 train_and_test_sets: list):

        if model_name not in {'Hy-MMSBM', 'HyperMOSBM'}:
            raise ValueError("model_name must be either 'Hy-MMSBM' or 'HyperMOSBM'.")

        self.H = H
        self.model_name = model_name
        self.train_and_test_sets = train_and_test_sets
        self.EPS = 1e-20
        self.K = K
        self.hs_clusters = hs_clusters

    def run(self):

        K = self.K

        lp_auc_lst = []
        lp_sample_size_lst = []
        runtime_lst = []

        for k in range(len(self.train_and_test_sets)):
            (
                H_train, H_test,
                (E_lp_pos, E_lp_neg),
            ) = self.train_and_test_sets[k]

            pos_scores_lp, neg_scores_lp = [], []

            t_s = time.time()

            if self.model_name == 'Hy-MMSBM':
                # model = hymmsbm.HyMMSBM(H_train, K)
                # _, (U, W) = model.fit()

                _, (U, W) = hymmsbm.fit_hymmsbm(H_train, K)

                pos_scores_lp = [hymmsbm.calc_hyperedge_score(e, U, W) for e in E_lp_pos]
                neg_scores_lp = [hymmsbm.calc_hyperedge_score(e, U, W) for e in E_lp_neg]

            elif self.model_name == 'HyperMOSBM':
                model = hypermosbm.HyperMOSBM(H_train, K, self.hs_clusters)
                _, (U, W) = model.fit()

                size_to_index = {}
                for l, s_set in enumerate(self.hs_clusters):
                    for s in s_set:
                        size_to_index[s] = l

                for e in E_lp_pos:
                    s = len(e)
                    if s in size_to_index:
                        l = size_to_index[s]
                        pos_scores_lp.append(hypermosbm.calc_hyperedge_score(e, U, W[l]))
                    else:
                        pos_scores_lp.append(0.0)
                for e in E_lp_neg:
                    s = len(e)
                    if s in size_to_index:
                        l = size_to_index[s]
                        neg_scores_lp.append(hypermosbm.calc_hyperedge_score(e, U, W[l]))
                    else:
                        neg_scores_lp.append(0.0)

            t_e = time.time()
            runtime_lst.append(t_e - t_s)

            auc = 0.0
            for pos_score, neg_score in zip(pos_scores_lp, neg_scores_lp):
                if pos_score > neg_score:
                    auc += 1.0
                elif math.isclose(pos_score, neg_score):
                    auc += 0.5

            if len(pos_scores_lp) > 0:
                auc = float(auc) / len(pos_scores_lp)
                lp_auc_lst.append(auc)
            else:
                lp_auc_lst.append(np.nan)

            lp_sample_size_lst.append(len(E_lp_pos))

        results = {
            'lp_auc': lp_auc_lst,
            'lp_sample_size': lp_sample_size_lst,
            'runtime': runtime_lst,
            'K': K,
        }

        return results

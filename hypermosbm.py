import numpy as np
import math
from scipy.sparse import csr_matrix
import hypergraph
import time
import warnings
import hymmsbm
from scipy.special import loggamma, gammaln

class HyperMOSBM:

    N = 0
    M = 0
    L = 0
    E = []
    A = []
    B = []
    K = 0
    U = []
    W = []
    poi_lambda = []
    S = []
    C = []
    C_for_U = []
    C_for_W = []
    D = 0
    hs_incidence = []
    size_to_index = {}
    test = False
    reg_lambda = 0
    regularization = None
    random_state = None
    normalizeU = False

    def __init__(self, H: hypergraph.HyperGraph, K: int, hs_clusters: list,
                 regularization=None, reg_lambda=0.0, test=False, normalizeU=False, random_state=None):

        hs_set = set(H.hyperedge_size_set())
        isolted_hs = hs_set - set([s for hs in hs_clusters for s in hs])
        #fake_hs = set([s for hs in hs_clusters for s in hs]) - hs_set

        if len(isolted_hs) > 0:
            warnings.warn(f'The following hyperedges are included in the input hypergraph '
                          f'but not appear in hs_clusters: {isolted_hs}')

        # if len(fake_hs) > 0:
        #     warnings.warn(f'The following hyperedges are included in hs_clusters '
        #                   f'but not appear in the input hypergraph: {fake_hs}')

        self.E = np.array([tuple(sorted(list(H.E[m]))) for m in range(0, len(H.E))], dtype=tuple)
        self.A = np.array([int(H.A[m]) for m in range(0, len(H.E))], dtype=int)
        self.N = H.N
        self.M = len(self.E)
        self.L = len(hs_clusters)
        self.K = K
        self.U = np.zeros((self.N, self.K), dtype=float)
        self.W = [np.zeros((self.K, self.K), dtype=float) for _ in range(0, self.L)]
        self.poi_lambda = np.zeros(self.M, dtype=float)
        self.S = [np.zeros((self.M, self.K), dtype=float) for _ in range(0, self.L)]
        self.C = np.zeros(self.L, dtype=float)
        self.C_for_U = [np.zeros((self.N, K), dtype=float) for _ in range(0, self.L)]
        self.C_for_W = [np.zeros((K, K), dtype=float) for _ in range(0, self.L)]
        self.D = max([len(e) for e in self.E])
        self.EPS = 1e-20
        self.hs_incidence = np.zeros((self.L, self.M), dtype=int)
        self.size_to_index = {}
        self.test = test
        self.reg_lambda = reg_lambda
        self.regularization = regularization
        self.random_state = random_state
        self.normalizeU = normalizeU

        for l in range(0, self.L):
            for s in hs_clusters[l]:
                self.size_to_index[s] = l

        coo_data_by_layer = [([], [], []) for _ in range(self.L)]
        for m in range(self.M):
            s = len(self.E[m])
            if s not in self.size_to_index:
                continue

            l = self.size_to_index[s]
            self.hs_incidence[l][m] = 1

            for i in self.E[m]:
                coo_data_by_layer[l][0].append(1)
                coo_data_by_layer[l][1].append(i)
                coo_data_by_layer[l][2].append(m)

        self.B = []
        for l in range(self.L):
            data, rows, cols = coo_data_by_layer[l]
            b_matrix = csr_matrix((data, (rows, cols)), shape=(self.N, self.M), dtype=int)
            self.B.append(b_matrix)

        # Constant C
        for l in range(0, self.L):
            self.C[l] = sum([float(2)/(s*(s-1)) for s in hs_clusters[l]])
            self.C_for_U[l] = np.full((self.N, self.K), self.C[l])
            self.C_for_W[l] = np.full((self.K, self.K), self.C[l])

        return

    def _initialize_params(self, random_state):
        rng = np.random.default_rng(random_state)

        # Matrix U
        self.U = rng.random((self.N, self.K))
        if self.normalizeU:
            self.U = self.U / self.U.sum(axis=1, keepdims=True)

        # Matrix W
        self.W = [rng.random((self.K, self.K)) for l in range(0, self.L)]
        for l in range(0, self.L):
            self.W[l] = np.triu(self.W[l], 0) + np.triu(self.W[l], 1).T

        # Matrix S
        for l in range(0, self.L):
            self.S[l] = self.B[l].transpose() @ self.U

        # lambda_e
        self.poi_lambda = np.zeros(self.M, dtype=float)
        for l in range(0, self.L):
            first_addend = ((self.S[l] @ self.W[l]) * self.S[l]).sum(axis=-1)
            second_addend = self.B[l].T @ (((self.U @ self.W[l]) * self.U).sum(axis=-1))
            self.poi_lambda += np.multiply(0.5 * (first_addend - second_addend), self.hs_incidence[l])
        self.poi_lambda = np.where(self.poi_lambda < self.EPS, self.EPS, self.poi_lambda)

        if self.test:
            for m in range(0, self.M):
                s = len(self.E[m])
                l = self.size_to_index[s]
                poi_lambda_ = 0.5 * np.sum([self.U[i][k]*self.U[j][q]*self.W[l][k][q] for i in self.E[m]
                                            for j in self.E[m] if j != i for k in range(0, self.K)
                                            for q in range(0, self.K)])
                if not math.isclose(self.poi_lambda[m], poi_lambda_, abs_tol=1e-20):
                    print("ERROR", m, self.poi_lambda[m], poi_lambda_)
                    exit()

        return

    def _check_initial_parameters(self):

        if np.any(np.isclose(self.U.sum(axis=1), 0)):
            # print(f"Error: sum of u_ik for some i is zero in layer {l}.")
            return False

        for l in range(self.L):
            if np.any(np.isclose(self.W[l], 0)):
                #print(f"Error: some w_kq is zero in layer {l}.")
                return False

            # if np.any(np.isclose(self.S[l].sum(axis=1), 0)):
            #     print(f"Error: sum of s_mk for some m is zero in layer {l}.")
            #     return False

        if np.any(np.isclose(self.poi_lambda, 0)):
            #print("Error: lambda_m for some m is zero.")
            return False

        return True

    def _update_u(self):

        num = np.zeros((self.N, self.K), dtype=float)

        for l in range(0, self.L):

            # Numerator
            multiplier =  np.multiply(self.A / (2.0 * self.poi_lambda), self.hs_incidence[l])
            weighting = self.B[l].multiply(multiplier[None, :])
            first_addend = weighting @ self.S[l]
            weighting_sum = np.asarray(weighting.sum(axis=1)).reshape(-1, 1)
            second_addend = weighting_sum * self.U
            num_l = 2.0 * (self.U * np.matmul(first_addend - second_addend, self.W[l]))

            if self.test:
                for i in range(0, self.N):
                    for k in range(0, self.K):
                        num_ = 0.0
                        for m in range(0, self.M):
                            if self.hs_incidence[l][m] != 1 or i not in self.E[m]:
                                continue
                            for j in self.E[m]:
                                if j == i:
                                    continue
                                for q in range(0, self.K):
                                    num_ += (self.A[m] * float(self.U[i][k]*self.U[j][q]*self.W[l][k][q])/
                                             self.poi_lambda[m])
                        if not math.isclose(num_l[i][k], num_, abs_tol=1e-20):
                            print("ERROR", i, k, num_l[i][k], num_)
                            exit()

            num += num_l

            # # Denominator
            # U_sum = self.U.sum(axis=0)
            # den_l = (self.C_for_U[l] * (np.matmul(self.W[l], U_sum)[None, :] - np.matmul(self.U, self.W[l])))
            # den_l = np.where(den_l < self.EPS, self.EPS, den_l)
            #
            # if self.test:
            #     for i in range(self.N):
            #         for k in range(self.K):
            #             den_ = 0.0
            #             x = 0.0
            #             for j in range(self.N):
            #                 if j == i:
            #                     continue
            #                 for q in range(self.K):
            #                     x += self.U[j][q] * self.W[l][q][k]
            #             den_ += x * self.C[l]
            #             if not math.isclose(den_l[i][k], den_, abs_tol=1e-20):
            #                 print("ERROR", i, k, den_l[i][k], den_)
            #                 exit()
            #
            # den += den_l

        # Denominator
        U_sum = self.U.sum(axis=0)

        W_avg = np.tensordot(self.C, np.array(self.W), axes=([0], [0]))

        den_part1 = np.matmul(W_avg, U_sum)
        den_part2 = np.matmul(self.U, W_avg)

        den = den_part1[None, :] - den_part2
        den = np.where(den < self.EPS, self.EPS, den)

        # Update U
        np.divide(num, den, out=self.U)

        if self.normalizeU:
            self.U[self.U < 0] = 0
            row_sums = self.U.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            self.U = self.U / row_sums

        # Matrix S
        for l in range(0, self.L):
            self.S[l] = self.B[l].transpose() @ self.U

        # lambda_e
        self.poi_lambda = np.zeros(self.M, dtype=float)
        for l in range(0, self.L):
            first_addend = ((self.S[l] @ self.W[l]) * self.S[l]).sum(axis=-1)
            second_addend = self.B[l].T @ (((self.U @ self.W[l]) * self.U).sum(axis=-1))
            self.poi_lambda += np.multiply(0.5 * (first_addend - second_addend), self.hs_incidence[l])
        self.poi_lambda = np.where(self.poi_lambda < self.EPS, self.EPS, self.poi_lambda)

        return

    def _update_w(self):

        u_sum = self.U.sum(axis=0)
        UTU = np.matmul(self.U.T, self.U)
        den_base = np.outer(u_sum, u_sum) - UTU

        for l in range(0, self.L):

            if self.regularization == None or self.regularization == 'l1':
                # Numerator
                multiplier =  np.multiply(self.A / (2.0 * self.poi_lambda), self.hs_incidence[l])
                first_addend = np.matmul(self.S[l].T, self.S[l] * multiplier[:, None])
                weighting = self.B[l].multiply(multiplier[None, :]).sum(axis=1)
                weighting = np.asarray(weighting).reshape(-1)
                second_addend = np.matmul(self.U.T, self.U * weighting[:, None])
                num = 2.0 * (self.W[l] * (first_addend - second_addend))

                if self.test:
                    for k in range(0, self.K):
                        for q in range(0, self.K):
                            num_ = 0.0
                            for m in range(0, self.M):
                                if self.hs_incidence[l][m] != 1:
                                    continue
                                for i in self.E[m]:
                                    for j in self.E[m]:
                                        if j == i:
                                            continue
                                        num_ += (self.A[m] * float(self.U[i][k]*self.U[j][q]*self.W[l][k][q])/
                                                 self.poi_lambda[m])
                            if not math.isclose(num[k][q], num_, abs_tol=1e-20):
                                print("ERROR", k, q, num[k][q], num_)
                                exit()

                # Denominator
                # u_sum = self.U.sum(axis=0)
                # den = (np.outer(u_sum, u_sum) - np.matmul(self.U.T, self.U)) * self.C_for_W[l] + self.reg_lambda
                # den = np.where(den < self.EPS, self.EPS, den)
                den = den_base * self.C[l] + self.reg_lambda
                den = np.where(den < self.EPS, self.EPS, den)

                if self.test:
                    for k in range(self.K):
                        for q in range(self.K):
                            den_ = 0.0
                            x = 0.0
                            for i in range(0, self.N):
                                for j in range(0, self.N):
                                    if j != i:
                                        x += self.U[i][k] * self.U[j][q]
                            den_ += x * self.C[l] + self.reg_lambda
                            if not math.isclose(den[k][q], den_, abs_tol=1e-20):
                                print("ERROR", k, q, den[k][q], den_)
                                exit()

                # Update W
                np.divide(num, den, out=self.W[l])
            elif self.regularization == 'l2':
                A = np.full((self.K, self.K), 2 * self.reg_lambda, dtype=float)

                u_sum = self.U.sum(axis=0)
                B = (np.outer(u_sum, u_sum) - np.matmul(self.U.T, self.U)) * self.C_for_W[l]
                #B = np.where(B < self.EPS, self.EPS, B)

                multiplier = np.multiply(self.A / (2.0 * self.poi_lambda), self.hs_incidence[l])
                first_addend = np.matmul(self.S[l].T, self.S[l] * multiplier[:, None])
                weighting = self.B[l].multiply(multiplier[None, :]).sum(axis=1)
                weighting = np.asarray(weighting).reshape(-1)
                second_addend = np.matmul(self.U.T, self.U * weighting[:, None])
                C = (-1) * 2.0 * (self.W[l] * (first_addend - second_addend))

                D = (B * B) - (4 * A * C)

                # Update W
                np.divide(((-1) * B + np.sqrt(D)), 2 * A, out=self.W[l])

        # lambda_e
        self.poi_lambda = np.zeros(self.M, dtype=float)
        for l in range(0, self.L):
            first_addend = ((self.S[l] @ self.W[l]) * self.S[l]).sum(axis=-1)
            second_addend = self.B[l].T @ (((self.U @ self.W[l]) * self.U).sum(axis=-1))
            self.poi_lambda += np.multiply(0.5 * (first_addend - second_addend), self.hs_incidence[l])
        self.poi_lambda = np.where(self.poi_lambda < self.EPS, self.EPS, self.poi_lambda)

        return

    def _calc_loglik(self):

        first_addend = 0.0
        U_sum = self.U.sum(axis=0)
        for l in range(0, self.L):
            first_addend += self.C[l] * 0.5 * (((U_sum @ self.W[l]) * U_sum).sum(axis=-1)
                                                      - ((self.U @ self.W[l]) * self.U).sum())
        second_addend = np.dot(self.A, np.log(self.poi_lambda))
        reg_term = self.reg_lambda * sum([np.sum(self.W[l]) for l in range(0, self.L)])

        if self.test:
            first_addend_ = 0.0
            for l in range(0, self.L):
                x = 0.0
                for i in range(0, self.N):
                    for j in range(0, self.N):
                        if j != i:
                            x += np.sum([self.U[i][k] * self.U[j][q] * self.W[l][k][q]
                                         for k in range(0, self.K) for q in range(0, self.K)])
                first_addend_ += 0.5 * self.C[l] * x

            if not math.isclose(first_addend, first_addend_, abs_tol=1e-20):
                print("ERROR", first_addend, first_addend_)
                exit()

            second_addend_ = 0.0
            for l in range(0, self.L):
                for m in range(0, self.M):
                    if self.hs_incidence[l][m] == 1:
                        second_addend_ += self.A[m] * np.log(self.poi_lambda[m])
            if not math.isclose(second_addend, second_addend_, abs_tol=1e-20):
                print("ERROR", second_addend, second_addend_)
                exit()

        return (-1) * first_addend + second_addend - reg_term


    def fit(self, n_init=10, max_iter=500, tol=None, check_convergence_every=1, threshold_for_convergence=1):

        best_loglik = float("-inf")
        best_param = None
        r_count = 0

        num_step_lst = []

        for i in range(0, n_init):
            if self.random_state == None:
                self._initialize_params(None)
            else:
                self._initialize_params(self.random_state + r_count)
            r_count += 1

            if self.random_state == None:
                self._initialize_params(None)
            else:
                while not self._check_initial_parameters():
                    self._initialize_params(self.random_state + r_count)
                    r_count += 1

            j = 0
            pre_loglik = float("-inf")

            converged = False
            n_tolerance_reached = 0

            while j < max_iter and converged == False:

                self._update_u()
                self._update_w()

                if j % check_convergence_every == 0:
                    L = self._calc_loglik()
                    
                    if tol is not None:
                        if abs(L - pre_loglik) < tol:
                            n_tolerance_reached += 1
                        else:
                            n_tolerance_reached = 0

                    pre_loglik = L

                if n_tolerance_reached > threshold_for_convergence:
                    converged = True

                j += 1

            L = self._calc_loglik()

            if L > best_loglik:
                best_loglik = L
                best_param = (self.U, self.W)

            num_step_lst.append(j)

        # print("Numbers of steps", num_step_lst)

        return best_loglik, best_param


    def calc_bic(self):

        loglik = self._calc_loglik()
        n_p = self.N * self.K + sum([float(self.K * (self.K + 1))/2 for _ in range(0, self.L)])

        return (-2) * loglik + n_p * math.log(self.N)

    def calc_test_loglik(self, H_test, U_trained, W_trained, full_likelihood=True):

        E_test = np.array([tuple(sorted(list(e))) for e in H_test.E], dtype=tuple)
        A_test = np.array([int(a) for a in H_test.A], dtype=int)
        M_test = len(E_test)

        coo_data_by_layer = [([], [], []) for _ in range(self.L)]
        hs_incidence_test = np.zeros((self.L, M_test), dtype=int)

        for m in range(M_test):
            s = len(E_test[m])
            if s not in self.size_to_index:
                continue
            l = self.size_to_index[s]
            hs_incidence_test[l][m] = 1
            for i in E_test[m]:
                coo_data_by_layer[l][0].append(1)
                coo_data_by_layer[l][1].append(i)
                coo_data_by_layer[l][2].append(m)

        B_test = []
        S_test = []
        for l in range(self.L):
            data, rows, cols = coo_data_by_layer[l]
            b_matrix = csr_matrix((data, (rows, cols)), shape=(self.N, M_test), dtype=int)
            B_test.append(b_matrix)
            S_test.append(b_matrix.transpose() @ U_trained)

        poi_lambda_test = np.zeros(M_test, dtype=float)
        for l in range(self.L):
            first = ((S_test[l] @ W_trained[l]) * S_test[l]).sum(axis=-1)
            second = B_test[l].T @ (((U_trained @ W_trained[l]) * U_trained).sum(axis=-1))
            poi_lambda_test += np.multiply(0.5 * (first - second), hs_incidence_test[l])

        poi_lambda_test = np.where(poi_lambda_test < self.EPS, self.EPS, poi_lambda_test)

        second_addend = np.dot(A_test, np.log(poi_lambda_test))

        first_addend = 0.0
        U_sum = U_trained.sum(axis=0)
        for l in range(self.L):
            first_addend += self.C[l] * 0.5 * (((U_sum @ W_trained[l]) * U_sum).sum(axis=-1)
                                               - ((U_trained @ W_trained[l]) * U_trained).sum())

        if full_likelihood:
            term_a = 0.0
            for m in range(len(E_test)):
                s = len(E_test[m])
                log_binom = gammaln(self.N - 1) - gammaln(s - 1) - gammaln(self.N - s + 1)
                log_kappa = math.log(s) + math.log(s - 1) - math.log(2) + log_binom
                term_a += A_test[m] * log_kappa

            A_test = np.array([int(a) for a in H_test.A], dtype=int)
            log_A_factorial_terms = np.array([math.lgamma(a + 1) for a in A_test])
            term_b = np.sum(log_A_factorial_terms)

            return (-1) * first_addend + second_addend - term_a - term_b
        else:
            return (-1) * first_addend + second_addend

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


class HyperlinkPrediction():

    H = None
    K = None
    hs_clusters = []
    size_to_index = {}
    EPS = 0
    train_and_test_sets = []

    def __init__(self, H: hypergraph.HyperGraph, hs_clusters: list, K: int, train_and_test_sets: list):
        self.H = H
        self.hs_clusters = hs_clusters
        self.K = K
        self.size_to_index = {}
        for l, s_set in enumerate(hs_clusters):
            for s in s_set:
                self.size_to_index[s] = l
        self.EPS = 1e-20
        self.train_and_test_sets = train_and_test_sets

    def calc(self):

        K = self.K

        auc_lst = []
        sample_size_lst = []
        runtime_lst = []

        for k in range(0, len(self.train_and_test_sets)):
            (
                H_train, H_test,
                (E_lp_pos, E_lp_neg),
            ) = self.train_and_test_sets[k]

            t_s = time.time()
            if len(self.hs_clusters) == 1:
                _, (U, W) = hymmsbm.fit_hymmsbm(H_train, K)
            else:
                model = HyperMOSBM(H_train, K, self.hs_clusters)
                _, (U, W) = model.fit()
            t_e = time.time()

            runtime_lst.append(t_e - t_s)

            pos_scores_lp = []
            neg_scores_lp = []
            for e in E_lp_pos:
                s = len(e)
                if len(self.hs_clusters) == 1:
                    pos_scores_lp.append(hymmsbm.calc_hyperedge_score(e, U, W))
                else:
                    if s in self.size_to_index:
                        l = self.size_to_index[s]
                        pos_scores_lp.append(calc_hyperedge_score(e, U, W[l]))
                    else:
                        pos_scores_lp.append(0.0)
            for e in E_lp_neg:
                s = len(e)
                if len(self.hs_clusters) == 1:
                    neg_scores_lp.append(hymmsbm.calc_hyperedge_score(e, U, W))
                else:
                    if s in self.size_to_index:
                        l = self.size_to_index[s]
                        neg_scores_lp.append(calc_hyperedge_score(e, U, W[l]))
                    else:
                        neg_scores_lp.append(0.0)

            auc_sum = 0.0
            for pos_score, neg_score in zip(pos_scores_lp, neg_scores_lp):
                if pos_score > neg_score:
                    auc_sum += 1.0
                elif math.isclose(pos_score, neg_score):
                    auc_sum += 0.5

            if len(pos_scores_lp) > 0:
                auc = float(auc_sum) / len(pos_scores_lp)
                auc_lst.append(auc)
            else:
                auc_lst.append(np.nan)

            sample_size_lst.append(H_test.M)

        return auc_lst, sample_size_lst, runtime_lst

class TestLogLikelihood():

    H = None
    K = None
    hs_clusters = []
    size_to_index = {}
    EPS = 0
    train_and_test_sets = []

    def __init__(self, H: hypergraph.HyperGraph, hs_clusters: list, K: int, train_and_test_sets: list):
        self.H = H
        self.hs_clusters = hs_clusters
        self.K = K
        self.size_to_index = {}
        for l, s_set in enumerate(hs_clusters):
            for s in s_set:
                self.size_to_index[s] = l
        self.EPS = 1e-20
        self.train_and_test_sets = train_and_test_sets

    def calc(self):

        K = self.K

        tll_lst = []
        sample_size_lst = []
        runtime_lst = []

        for k in range(0, len(self.train_and_test_sets)):
            H_train, H_test = self.train_and_test_sets[k][0], self.train_and_test_sets[k][1]

            t_s = time.time()
            model = HyperMOSBM(H_train, K, self.hs_clusters)
            best_loglik, (U, W) = model.fit()
            t_e = time.time()

            runtime_lst.append(t_e - t_s)

            tll = model.calc_test_loglik(H_test, U, W)
            tll_lst.append(tll)

            sample_size_lst.append(H_test.M)

        return tll_lst, sample_size_lst, runtime_lst


def partition_hyperedge_size_set(H: hypergraph.HyperGraph, K: int, train_and_test_sets: list,
                                 model_selection: str = 'auc'):

    if model_selection not in {'tll', 'auc'}:
        print("ERROR: model_selection must be either 'tll' or 'auc'.")
        return [], []

    hs_set_ = sorted(list(H.hyperedge_size_set()))
    s_min, s_max = 2, max(hs_set_)
    hs_set = range(s_min, s_max + 1)

    if len(hs_set) <= 4 and model_selection == 'tll':
        res_history, hs_clusters = _exhaustive_search_by_tll(H, K, train_and_test_sets)
        return res_history, hs_clusters
    elif len(hs_set) <= 4 and model_selection == 'auc':
        res_history, hs_clusters = _exhaustive_search_by_auc(H, K, train_and_test_sets)
        return res_history, hs_clusters
    elif len(hs_set) > 4 and model_selection == 'tll':
        res_history, hs_clusters = _greedy_search_by_tll(H, K, train_and_test_sets)
        return res_history, hs_clusters
    elif len(hs_set) > 4 and model_selection == 'auc':
        res_history, hs_clusters = _greedy_search_by_auc(H, K, train_and_test_sets)
        return res_history, hs_clusters


def _greedy_search_by_tll(H: hypergraph.HyperGraph, K: int, train_and_test_sets: list,
                          diff_threshold=1.0, c_factor=5):

    hs_set = H.hyperedge_size_set()
    s_min, s_max = 2, max(hs_set)
    hs_clusters = [sorted(list(range(s_min, s_max+1)))]

    num_params = K * (K + 1) / 2
    min_sample_threshold = c_factor * num_params

    counts_by_size = {s: 0 for s in range(s_min, s_max + 1)}
    for e in H.E:
        s = len(e)
        if s in counts_by_size:
            counts_by_size[s] += 1

    TLL = TestLogLikelihood(H, hs_clusters, K, train_and_test_sets)
    tll_lst, sample_size_lst, runtime_lst = TLL.calc()
    best_tll_list = tll_lst
    res_history = [[hs_clusters, tll_lst, sample_size_lst, runtime_lst]]

    flag = True
    while flag:

        baseline_median_tll = np.median(np.array(best_tll_list))

        found_any_improvement_in_this_round = False
        best_split_tll_median_in_round = -np.inf
        best_split_config_in_round = None
        best_split_results_in_round = None

        for cluster_idx, cluster_to_split in enumerate(hs_clusters):
            if len(cluster_to_split) < 2:
                continue

            for split_point_offset in range(len(cluster_to_split) - 1):
                new_cluster_part1 = sorted(cluster_to_split[:split_point_offset + 1])
                new_cluster_part2 = sorted(cluster_to_split[split_point_offset + 1:])

                count1 = sum(counts_by_size.get(s, 0) for s in new_cluster_part1)
                count2 = sum(counts_by_size.get(s, 0) for s in new_cluster_part2)

                if count1 < min_sample_threshold or count2 < min_sample_threshold:
                    continue

                temp_hs_clusters = []
                for i, c in enumerate(hs_clusters):
                    if i == cluster_idx:
                        temp_hs_clusters.append(new_cluster_part1)
                        temp_hs_clusters.append(new_cluster_part2)
                    else:
                        temp_hs_clusters.append(c)

                temp_hs_clusters_sorted = sorted([sorted(c) for c in temp_hs_clusters])

                TLL_candidate = TestLogLikelihood(H, temp_hs_clusters_sorted, K, train_and_test_sets)
                tll_lst_candidate, sample_size_lst_candidate, runtime_lst_candidate = TLL_candidate.calc()

                current_split_median = np.median(np.array(tll_lst_candidate))

                diff = current_split_median - baseline_median_tll

                if diff > diff_threshold and current_split_median > best_split_tll_median_in_round:
                    best_split_tll_median_in_round = current_split_median
                    best_split_config_in_round = temp_hs_clusters_sorted
                    best_split_results_in_round = [tll_lst_candidate,
                                                   sample_size_lst_candidate,
                                                   runtime_lst_candidate]
                    found_any_improvement_in_this_round = True

        if found_any_improvement_in_this_round:
            best_tll_list = best_split_results_in_round[0]
            hs_clusters = best_split_config_in_round
            res_history.append([best_split_config_in_round] + best_split_results_in_round)
        else:
            flag = False

        if all(len(c) == 1 for c in hs_clusters):
            flag = False

    return res_history, hs_clusters


def _greedy_search_by_auc(H: hypergraph.HyperGraph, K: int, train_and_test_sets: list,
                          diff_threshold=0.001, c_factor=5):

    hs_set = H.hyperedge_size_set()
    s_min, s_max = 2, max(hs_set)
    hs_clusters = [sorted(list(range(s_min, s_max + 1)))]

    num_params = K * (K + 1) / 2
    min_sample_threshold = c_factor * num_params

    counts_by_size = {s: 0 for s in range(s_min, s_max + 1)}
    for e in H.E:
        s = len(e)
        if s in counts_by_size:
            counts_by_size[s] += 1

    LP = HyperlinkPrediction(H, hs_clusters, K, train_and_test_sets)
    lp_auc_lst, sample_size_lst, runtime_lst = LP.calc()
    best_auc_list = lp_auc_lst
    res_history = [[hs_clusters, lp_auc_lst, sample_size_lst, runtime_lst]]

    flag = True
    while flag:

        baseline_mean_auc = np.nanmean(np.array(best_auc_list))

        found_any_improvement_in_this_round = False
        best_split_auc_mean_in_round = -np.inf
        best_split_config_in_round = None
        best_split_results_in_round = None

        for cluster_idx, cluster_to_split in enumerate(hs_clusters):
            if len(cluster_to_split) < 2:
                continue

            for split_point_offset in range(len(cluster_to_split) - 1):
                new_cluster_part1 = sorted(cluster_to_split[:split_point_offset + 1])
                new_cluster_part2 = sorted(cluster_to_split[split_point_offset + 1:])

                count1 = sum(counts_by_size.get(s, 0) for s in new_cluster_part1)
                count2 = sum(counts_by_size.get(s, 0) for s in new_cluster_part2)

                if count1 < min_sample_threshold or count2 < min_sample_threshold:
                    continue

                temp_hs_clusters = []
                for i, c in enumerate(hs_clusters):
                    if i == cluster_idx:
                        temp_hs_clusters.append(new_cluster_part1)
                        temp_hs_clusters.append(new_cluster_part2)
                    else:
                        temp_hs_clusters.append(c)

                temp_hs_clusters_sorted = sorted([sorted(c) for c in temp_hs_clusters])

                AUC_candidate = HyperlinkPrediction(H, temp_hs_clusters_sorted, K, train_and_test_sets)
                auc_lst_candidate, sample_size_lst_candidate, runtime_lst_candidate = AUC_candidate.calc()

                current_split_mean = np.nanmean(np.array(auc_lst_candidate))

                diff = float(current_split_mean - baseline_mean_auc)

                if diff > diff_threshold and current_split_mean > best_split_auc_mean_in_round:
                    best_split_auc_mean_in_round = current_split_mean
                    best_split_config_in_round = temp_hs_clusters_sorted
                    best_split_results_in_round = [auc_lst_candidate, sample_size_lst_candidate,
                                                   runtime_lst_candidate]
                    found_any_improvement_in_this_round = True

        if found_any_improvement_in_this_round:
            best_auc_list = best_split_results_in_round[0]
            hs_clusters = best_split_config_in_round
            res_history.append([best_split_config_in_round] + best_split_results_in_round)
        else:
            flag = False

        if all(len(c) == 1 for c in hs_clusters):
            flag = False

    return res_history, hs_clusters


def _generate_partitions(elements):

    if not elements:
        yield []
        return

    first = elements[0]
    rest = elements[1:]

    for p in _generate_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]

        yield [[first]] + p


def _exhaustive_search_by_tll(H: hypergraph.HyperGraph, K: int, train_and_test_sets: list,
                              max_n_for_exhaustive=4, c_factor=5):

    hs_set_ = sorted(list(H.hyperedge_size_set()))
    s_min, s_max = 2, max(hs_set_)
    hs_set = list(range(s_min, s_max + 1))
    n_sizes = len(hs_set)

    if n_sizes > max_n_for_exhaustive:
        print(
            f"Warning: Number of unique hyperedge sizes ({n_sizes}) is larger than the threshold ({max_n_for_exhaustive}).")
        print("Exhaustive search is infeasible and will be skipped.")
        return [], None

    num_params = K * (K + 1) / 2
    min_sample_threshold = c_factor * num_params

    counts_by_size = {s: 0 for s in range(s_min, s_max + 1)}
    for e in H.E:
        s = len(e)
        if s in counts_by_size:
            counts_by_size[s] += 1

    all_partitions = list(_generate_partitions(hs_set))

    best_tll_median = -np.inf
    best_partition = None
    res_history = []

    for i, partition_candidate in enumerate(all_partitions):
        hs_clusters = sorted([sorted(c) for c in partition_candidate])

        is_valid_partition = all(
            sum(counts_by_size.get(s, 0) for s in subset) >= min_sample_threshold
            for subset in hs_clusters
        )

        if not is_valid_partition:
            continue

        TLL_evaluator = TestLogLikelihood(H, hs_clusters, K, train_and_test_sets)
        tll_lst, sample_size_lst, runtime_lst = TLL_evaluator.calc()

        current_tll_median = np.median(np.array(tll_lst))

        res_history.append([hs_clusters, tll_lst, sample_size_lst, runtime_lst])

        if current_tll_median > best_tll_median:
            best_tll_median = current_tll_median
            best_partition = hs_clusters
            #print(f"  -> New best partition found with median TLL: {best_tll_median:.4f}")

    return res_history, best_partition


def _exhaustive_search_by_auc(H: hypergraph.HyperGraph, K: int, train_and_test_sets: list,
                              max_n_for_exhaustive=4, c_factor=5):

    hs_set_ = sorted(list(H.hyperedge_size_set()))
    s_min, s_max = 2, max(hs_set_)
    hs_set = list(range(s_min, s_max + 1))
    n_sizes = len(hs_set)

    if n_sizes > max_n_for_exhaustive:
        #warnings.warn(f"...")
        return [], None

    num_params = K * (K + 1) / 2
    min_sample_threshold = c_factor * num_params

    counts_by_size = {s: 0 for s in range(s_min, s_max + 1)}
    for e in H.E:
        s = len(e)
        if s in counts_by_size:
            counts_by_size[s] += 1

    all_partitions = list(_generate_partitions(hs_set))
    best_auc_mean = -np.inf
    best_partition = None
    res_history = []

    #print(f"Starting exhaustive search over {len(all_partitions)} partitions...")

    for i, partition_candidate in enumerate(all_partitions):
        hs_clusters = sorted([sorted(c) for c in partition_candidate])

        is_valid_partition = all(
            sum(counts_by_size.get(s, 0) for s in subset) >= min_sample_threshold
            for subset in hs_clusters
        )

        if not is_valid_partition:
            continue

        LP_evaluator = HyperlinkPrediction(H, hs_clusters, K, train_and_test_sets)
        lp_auc_lst, sample_size_lst, runtime_lst = LP_evaluator.calc()

        current_auc_mean = np.nanmean(lp_auc_lst)

        res_history.append([hs_clusters, lp_auc_lst, sample_size_lst, runtime_lst])

        if current_auc_mean > best_auc_mean:
            best_auc_mean = current_auc_mean
            best_partition = hs_clusters
            #print(f"    New best partition found with mean AUC = {best_auc_mean:.4f}")

    print(f"Exhaustive search finished. Best partition: {best_partition} with mean AUC = {best_auc_mean:.4f}")

    return res_history, best_partition

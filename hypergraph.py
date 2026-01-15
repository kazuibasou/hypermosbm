import math
import warnings
import numpy as np
from collections import Counter
import pickle
from collections import deque
from collections import defaultdict
import xgi
from datetime import datetime
import random
import collections
import os.path as osp
from scipy.sparse import load_npz, coo_matrix
import json
from itertools import combinations

data_dir = "./data/"

class HyperGraph():
    N = 0
    M = 0
    E = []
    A = np.zeros(M, dtype=int)
    X = np.zeros(N, dtype=int)
    Z = 0

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.E = [tuple() for _ in range(0, self.M)]
        self.A = np.zeros(self.M, dtype=int)
        self.X = np.zeros(self.N, dtype=int)
        self.Z = 0

    def is_connected(self):
        vlist = defaultdict(list)
        elist = defaultdict(list)
        for m in range(0, self.M):
            for i in self.E[m]:
                vlist[m].append(i)
                elist[i].append(m)

        searched = {i: 0 for i in range(0, self.N)}
        nodes = set()
        v = 0

        Q = deque()
        searched[v] = 1
        Q.append(v)
        while len(Q) > 0:
            v = Q.popleft()
            nodes.add(v)
            for m in elist[v]:
                for w in vlist[m]:
                    if searched[w] == 0:
                        searched[w] = 1
                        Q.append(w)

        return self.N == sum(list(searched.values()))


    def calc_pairwise_shortest_path_length(self):
        vlist = defaultdict(list)
        elist = defaultdict(list)
        for m in range(0, self.M):
            for i in self.E[m]:
                vlist[m].append(i)
                elist[i].append(m)

        d = np.full((self.N, self.N), -1)

        for s in range(0, self.N):
            Q = deque()
            d[s][s] = 0
            Q.append(s)

            while len(Q) > 0:
                v = Q.popleft()
                for m in elist[v]:
                    for w in vlist[m]:
                        if d[s][w] < 0:
                            d[s][w] = d[s][v] + 1
                            Q.append(w)

        return d

    def hyperedge_size_set(self):
        hs_set = []
        for m in range(0, self.M):
            hs_set.append(len(self.E[m]))

        return set(hs_set)

    def hyperedge_size(self):
        hs = {}
        for m in range(0, self.M):
            hs[m] = len(self.E[m])

        return hs

    def hyperedge_size_distribution(self, func):
        if func == 'freq':
            return dict(Counter(list(self.hyperedge_size().values())))
        elif func == 'prob':
            d = dict(Counter(list(self.hyperedge_size().values())))
            return {k: float(d[k])/self.N for k in d}
        elif func == 'survival':
            d_ = dict(Counter(list(self.hyperedge_size().values())))
            lst = sorted(list(d_.items()), key=lambda x:x[0], reverse=False)
            d = {}
            for i in range(0, len(lst)):
                k = lst[i][0]
                n = sum([lst[j][1] for j in range(i, len(lst))])
                d[k] = float(n)/self.M
            return d
        else:
            print("ERROR: given function is not supported.")
            return {}

    def s_size_hyperedge_induced_hypergraph(self, s_set: set):

        E_ = [e for e in self.E if len(e) in s_set]
        A_ = [self.A[m] for m in range(0, self.M) if len(self.E[m]) in s_set]
        H_ = HyperGraph(self.N, len(E_))
        H_.E = E_
        H_.A = A_

        return H_

    def get_largest_connected_component(self):

        vlist = defaultdict(list)
        elist = defaultdict(list)
        for m in range(0, self.M):
            for i in self.E[m]:
                vlist[m].append(i)
                elist[i].append(m)

        visited_nodes = {i: False for i in range(self.N)}
        visited_hyperedges = {m: False for m in range(self.M)}

        largest_component_nodes = set()
        largest_component_hyperedges = set()
        max_component_size = 0

        for start_node_id in range(self.N):
            if not visited_nodes[start_node_id] and start_node_id in elist:

                current_component_nodes = set()
                current_component_hyperedges = set()
                q = deque()

                visited_nodes[start_node_id] = True
                q.append(start_node_id)
                current_component_nodes.add(start_node_id)

                while q:
                    v = q.popleft()

                    for m_id in elist[v]:
                        if not visited_hyperedges[m_id]:
                            visited_hyperedges[m_id] = True
                            current_component_hyperedges.add(m_id)

                            for w_id in vlist[m_id]:
                                if not visited_nodes[w_id]:
                                    visited_nodes[w_id] = True
                                    q.append(w_id)
                                    current_component_nodes.add(w_id)

                if len(current_component_nodes) > max_component_size:
                    max_component_size = len(current_component_nodes)
                    largest_component_nodes = current_component_nodes
                    largest_component_hyperedges = current_component_hyperedges

        if not largest_component_nodes and self.N > 0:
            pass

        V = [i for i in largest_component_nodes]
        E = [self.E[m] for m in largest_component_hyperedges]
        A = [self.A[m] for m in largest_component_hyperedges]

        return V, E, A

    def projected_graph(self):

        A_ = {}
        for m in range(0, len(self.E)):
            e = tuple(sorted(list(self.E[m])))
            for (i, j) in combinations(e, 2):
                e_ = tuple(sorted((i, j)))
                if e_ not in A_:
                    A_[e_] = 1
                else:
                    A_[e_] += 1

        E_ = list(A_.keys())

        G = HyperGraph(self.N, len(E_))
        G.E = np.array(E_, dtype=tuple)
        G.A = np.array([int(A_[e]) for e in E_], dtype=int)

        return G

def read_hypergraph(data_name, multiple_hyperedges, print_info=True):

    if data_name in {"house-committees", "senate-committees", "justice", "walmart"}:
        H = read_nicolo_hypergraph_data(data_name, multiple_hyperedges, print_info)
    elif data_name in {"contact-primary-school", "contact-high-school"}:
        H = read_benson_hypergraph_data(data_name, multiple_hyperedges, print_info)
    else:
        print("ERROR: given data set is not defined.")
        exit()

    return H

def read_benson_hypergraph_data(data_name, multiple_hyperedges, print_info=True):

    if data_name not in {"contact-primary-school", "contact-high-school"}:
        f1_path = data_dir + str(data_name) + "/" + str(data_name) + "_nverts.txt"
        f1 = open(f1_path, 'r')
        f2_path = data_dir + str(data_name) + "/" + str(data_name) + "_hyperedges.txt"
        f2 = open(f2_path, 'r')

        lines1 = f1.readlines()
        lines2 = f2.readlines()

        E, A, V = [], [], []
        c = 0
        for line1 in lines1:
            nv = int(line1[:-1].split(" ")[0])

            e = []
            for i in range(0, nv):
                v = int(lines2[c + i][:-1])
                e.append(v)

            e = tuple(sorted(list(set(e))))
            if len(e) < 2:
                continue

            if e not in E:
                E.append(e)
                A.append(1)
            elif multiple_hyperedges:
                m = E.index(e)
                A[m] += 1

            V += list(e)
            c += nv

        f1.close()
        f2.close()

        V = list(sorted(list(set(V))))
        if V != list(range(0, len(V))):
            print("ERROR: node indices are not valid.")
            print(set(range(0, len(V))) - set(V), set(V) - set(range(0, len(V))))
            exit()

        N = len(V)
        M = len(E)

        H = HyperGraph(N, M)
        H.E = list(E)
        H.A = np.array(A)
    else:
        f1_path = data_dir + str(data_name) + "/hyperedges-" + str(data_name) + ".txt"
        f1 = open(f1_path, 'r')
        f2_path = data_dir + str(data_name) + "/node-labels-" + str(data_name) + ".txt"
        f2 = open(f2_path, 'r')

        lines1 = f1.readlines()
        lines2 = f2.readlines()

        E, A, V = [], [], []
        c = 0
        for line1 in lines1:
            e = tuple(sorted(list(set([int(v) - 1 for v in line1[:-1].split(",")]))))

            if len(e) < 2:
                continue

            if e not in E:
                E.append(e)
                A.append(1)
            elif multiple_hyperedges:
                m = E.index(e)
                A[m] += 1

            V += list(e)

        node_label = {}
        i = 0
        for line2 in lines2:
            x = int(line2[:-1]) - 1
            node_label[i] = x
            i += 1

        f1.close()
        f2.close()

        V = list(sorted(list(set(V))))
        if V != list(range(0, len(V))):
            print("ERROR: node indices are not valid.")
            print(set(range(0, len(V))) - set(V), set(V) - set(range(0, len(V))))
            exit()

        N = len(V)
        M = len(E)

        H = HyperGraph(N, M)
        H.E = list(E)
        H.A = np.array(A)
        H.X = np.array([node_label[i] for i in range(0, N)])
        H.Z = len(set(list(H.X)))

    if print_info:
        sum_degree = sum([len(H.E[m]) for m in range(0, H.M)])
        D = max([len(H.E[m]) for m in range(0, H.M)])
        count_by_size = {}
        for m in range(0, M):
            s = len(H.E[m])
            count_by_size[s] = count_by_size.get(s, 0) + 1

        print("Number of nodes: " + str(H.N))
        print("Number of hyperedges: " + str(H.M))
        print("Sum of hyperedge weights: " + str(np.sum(H.A)))
        print("Average degree of the node: " + str(float(sum_degree) / H.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / H.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Hyperedge size distribution: " + str(list(sorted(count_by_size.items()))))
        print("Connected hypergraph: " + str(H.is_connected()))
        print()

    return H


def read_nicolo_hypergraph_data(data_name, multiple_hyperedges, print_info=True):
    ori_data_name = {
        "contact-primary-school": "primary_school",
        "contact-high-school": "high_school",
        "workplace": "workspace1",
        "hospital": "hospital",
        "gene-disease": "curated_gene_disease_associations",
        "house-committees": "house-committees",
        "senate-committees": "senate-committees",
        "justice": "justice",
        "walmart": "walmart-trips_3core",
        "enron-email": "enron-email",
        "trivago": "trivago-clicks_5core",
        "house-bills": "house-bills",
        "senate-bills": "senate-bills",
    }

    if data_name not in ori_data_name:
        print("Error: given data set is not found.")
        exit()

    f_path = data_dir + data_name + "/" + ori_data_name[data_name] + ".npz"
    npzfile = np.load(f_path, allow_pickle=True)
    A, B, hyperedges = npzfile['A'], npzfile['B'], npzfile['hyperedges']

    if len([e for e in hyperedges if len(set(e)) < 2]) > 0:
        print("Error: hyperedges with |e| < 2 are included.")
        exit()

    V = []
    for e in hyperedges:
        V += list(e)

    V = list(sorted(list(set(V))))
    if V != list(range(0, len(V))):
        print("ERROR: node indices are not valid.")
        print(set(range(0, len(V))) - set(V), set(V) - set(range(0, len(V))))
        exit()

    if data_name == 'enron-email':
        M = len(hyperedges)
        N = len(V)
    else:
        N, M = int(B.shape[0]), len(hyperedges)

    H = HyperGraph(N, M)
    H.E = [tuple(sorted(list(e))) for e in hyperedges if len(tuple(sorted(list(e)))) >= 2]
    if multiple_hyperedges:
        H.A = np.array(A)
    else:
        H.A = np.ones(len(A), dtype=float)

    if print_info:
        sum_degree = sum([len(H.E[m]) for m in range(0, H.M)])
        D = max([len(H.E[m]) for m in range(0, H.M)])
        count_by_size = {}
        for m in range(0, M):
            s = len(H.E[m])
            count_by_size[s] = count_by_size.get(s, 0) + 1

        print("Number of nodes: " + str(H.N))
        print("Number of hyperedges: " + str(H.M))
        print("Sum of hyperedge weights: " + str(np.sum(H.A)))
        print("Average degree of the node: " + str(float(sum_degree) / H.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / H.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Hyperedge size distribution: " + str(list(sorted(count_by_size.items()))))
        print("Connected hypergraph: " + str(H.is_connected()))
        print()

    return H


def read_hypergraph_from_xgi(data_name, multiple_hyperedges, print_info):

    H = xgi.load_xgi_data(data_name)

    lst = list(H.edges.members(dtype=dict).items())

    E, A, V = [], [], []
    hyperedge_index = {}
    for (i, e) in lst:
        e = tuple(sorted(list(int(v) for v in e)))

        if len(e) < 2:
            continue

        if e not in hyperedge_index:
            E.append(e)
            A.append(1)
            m = len(hyperedge_index)
            hyperedge_index[e] = m
        elif multiple_hyperedges:
            m = hyperedge_index[e]
            A[m] += 1

        V += list(e)

    V = list(sorted(list(set(V))))
    node_index_map = {}
    for i in range(0, len(V)):
        node_index_map[V[i]] = i

    V = [node_index_map[v] for v in V]
    E = [tuple(sorted(list([node_index_map[v] for v in e]))) for e in E]

    N = len(V)
    M = len(E)

    H = HyperGraph(N, M)
    H.E = list(E)
    H.A = np.array(A)

    if print_info:
        sum_degree = sum([len(H.E[m]) for m in range(0, H.M)])
        D = max([len(H.E[m]) for m in range(0, H.M)])
        count_by_size = {}
        for m in range(0, M):
            s = len(H.E[m])
            count_by_size[s] = count_by_size.get(s, 0) + 1

        print("Data name: " + str(data_name))
        print("Number of nodes: " + str(H.N))
        print("Number of hyperedges: " + str(H.M))
        print("Sum of hyperedge weights: " + str(np.sum(H.A)))
        print("Average degree of the node: " + str(float(sum_degree) / H.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / H.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Hyperedge size distribution: " + str(list(sorted(count_by_size.items()))))
        print("Connected hypergraph: " + str(H.is_connected()))
        print()

    return H


def construct_hypergraphs(multiple_hyperedges):

    for hypergraph_name in ["hospital-lyon", "invs13", "invs15"]:

        print(hypergraph_name)
        H = read_hypergraph_from_xgi(hypergraph_name, multiple_hyperedges, print_info=True)

        f_path = './data/' + str(hypergraph_name) + '.pickle'
        with open(f_path, mode='wb') as f:
            pickle.dump(H, f)

    for hypergraph_name in ["contact-primary-school", "contact-high-school", "house-committees", "senate-committees", "justice", "walmart"]:

        print(hypergraph_name)
        H = read_hypergraph(hypergraph_name, multiple_hyperedges, print_info=True)

        f_path = './data/' + str(hypergraph_name) + '.pickle'
        with open(f_path, mode='wb') as f:
            pickle.dump(H, f)

    return


def read_pickle_file(f_name):

    f_path = data_dir + f_name
    with open(f_path, mode="rb") as f:
        d = pickle.load(f)

    return d


def construct_cocitation_hypergraph(field, field_name, multiple_hyperedges, top_prop, connected, min_pub_year,
                                    max_pub_year, min_cocitation_freq, max_hyperedge_size,
                                    min_papers_per_subfield, print_info=True):

    f_name = f'openalex_work_data_{field}.pickle'
    work_data = read_pickle_file(f_name)

    w_ids = set(work_data.keys())
    print(f"Number of papers in {field}:", len(w_ids))

    topic_data = read_pickle_file('openalex_topic_data.pickle')
    subfields = set()
    for t_id in topic_data:
        field_ = topic_data[t_id]["field"]["display_name"]
        if field_ == field:
            subfield = topic_data[t_id]["subfield"]["display_name"]
            subfields.add(subfield)

    print("Number of subfields:", len(subfields))

    invalid_titles = {"none", "", "deleted work"}

    work_subfield = {}
    pub_ys = []
    for w_id in work_data:
        pub_y = int(datetime.strptime(str(work_data[w_id]["pub_d"]), '%Y-%m-%d').year)
        pub_ys.append(pub_y)
        if pub_y < min_pub_year or pub_y > max_pub_year:
            continue

        title = work_data[w_id]["title"]
        if len(title) <= 0 or title in invalid_titles:
            continue

        t_id = work_data[w_id]["t_id"]
        sf = topic_data[t_id]["subfield"]["display_name"]

        if sf not in subfields:
            continue

        work_subfield[w_id] = sf

    work_cited = defaultdict(int)
    for w_id in work_data:
        for w_id_ in set(work_data[w_id]["refs"]) & w_ids:
            work_cited[w_id_] += 1

    print("Publication year range in the original data:", min(pub_ys), max(pub_ys))

    works_by_sf = defaultdict(list)
    for w_id in work_subfield:
        sf = work_subfield[w_id]
        if sf not in subfields:
            continue
        works_by_sf[sf].append((w_id, work_cited[w_id]))

    # top_works = []
    # num_top_works_by_sf = defaultdict(int)
    # for sf in subfields:
    #     lst = [paper[0] for paper in list(sorted(works_by_sf[sf], key=lambda x: x[1], reverse=True))]
    #     top_count = int(len(lst) * top_prop)
    #     top_works += lst[:top_count]
    #     num_top_works_by_sf[sf] = top_count
    # top_works = set(top_works)

    top_works = []
    num_top_works_by_sf = defaultdict(int)
    cumulative_citation_prop = top_prop

    for sf in subfields:
        sorted_papers = sorted(works_by_sf[sf], key=lambda x: x[1], reverse=True)

        if not sorted_papers:
            continue

        total_citations_in_sf = sum(paper[1] for paper in sorted_papers)
        citation_threshold = total_citations_in_sf * cumulative_citation_prop

        selected_papers_for_sf = []
        current_cumulative_citations = 0
        for paper_id, cited_count in sorted_papers:
            selected_papers_for_sf.append(paper_id)
            current_cumulative_citations += cited_count
            if current_cumulative_citations >= citation_threshold:
                break

        if len(selected_papers_for_sf) < min_papers_per_subfield and len(sorted_papers) >= min_papers_per_subfield:
            selected_papers_for_sf = [paper[0] for paper in sorted_papers[:min_papers_per_subfield]]

        top_works.extend(selected_papers_for_sf)
        num_top_works_by_sf[sf] = len(selected_papers_for_sf)

    top_works = set(top_works)

    print("Number of top works: " + str(len(top_works)))
    print("Number of top works by subfield:", num_top_works_by_sf)

    paper_info = {}
    for w_id in work_data:
        if w_id not in top_works:
            continue

        paper_info[w_id] = {
            "id": w_id,
            "title": work_data[w_id]["title"],
            "pub_d": work_data[w_id]["pub_d"],
            "primary_topic": topic_data[work_data[w_id]["t_id"]],
            "cited_by_count": int(work_cited[w_id]),
        }

    cocitation_freq = {}
    for w_id in work_data:
        if w_id in top_works:
            continue

        pub_y = int(datetime.strptime(str(work_data[w_id]["pub_d"]), '%Y-%m-%d').year)
        if pub_y < min_pub_year or pub_y > max_pub_year:
            continue

        cited_works = set(work_data[w_id]["refs"]) & set(top_works)

        if len(cited_works) < 2 or len(cited_works) > max_hyperedge_size:
            continue

        e = tuple(sorted(list([p_id_ for p_id_ in cited_works])))

        if e not in cocitation_freq:
            cocitation_freq[e] = 1
        else:
            cocitation_freq[e] += 1

    subfield_lst = sorted(list(subfields))
    subfield_index = {subfield_lst[i]: i for i in range(0, len(subfield_lst))}

    E, A, V = [], [], []
    hyperedge_index = {}
    for cited_works in cocitation_freq:
        if cocitation_freq[cited_works] < min_cocitation_freq:
            continue

        e = []
        for p_id in cited_works:
            e.append(p_id)

        e = tuple(sorted(list([str(v) for v in e])))

        if len(e) < 2 or len(e) > max_hyperedge_size:
            continue

        if e not in hyperedge_index:
            E.append(e)
            m = len(hyperedge_index)
            hyperedge_index[e] = m

        if multiple_hyperedges:
            A.append(cocitation_freq[cited_works])
        else:
            A.append(1)

        V += list(e)

    V = list(sorted(list(set(V))))
    node_index_to_paper_id = {}
    paper_id_to_node_index = {}
    node_index_to_subfield_index = {}
    for i in range(0, len(V)):
        p_id = V[i]
        node_index_to_paper_id[i] = p_id
        paper_id_to_node_index[p_id] = i
        node_index_to_subfield_index[i] = subfield_index[work_subfield[p_id]]

    V = [i for i in range(0, len(V))]
    E = [tuple(sorted(list([paper_id_to_node_index[p_id] for p_id in e]))) for e in E]
    X = [node_index_to_subfield_index[i] for i in V]

    N = len(V)
    M = len(E)

    H = HyperGraph(N, M)
    H.E = list(E)
    H.A = np.array(A)
    H.X = np.array(X)
    H.Z = len(set(list(H.X)))
    H.paper_info = {i: paper_info[node_index_to_paper_id[i]] for i in V}

    print("Original hypergraph", N, M)

    if connected:
        V, E, A = H.get_largest_connected_component()
        V = list(sorted(list(V)))
        node_index_map = {}
        for i in range(0, len(V)):
            node_index_map[V[i]] = i
        V = [node_index_map[v] for v in V]
        E = [tuple(sorted(list([node_index_map[v] for v in e]))) for e in E]
        X = [H.X[v] for v in V]

        N = len(V)
        M = len(E)

        H = HyperGraph(N, M)
        H.E = list(E)
        H.A = np.array(A)
        H.X = np.array(X)
        H.Z = len(set(list(H.X)))

        print("Largest connected component", N, M)

    f_path = f'./data/{field_name}-cocitations.pickle'
    with open(f_path, mode='wb') as f:
        pickle.dump(H, f)

    if print_info:
        sum_degree = sum([len(H.E[m]) for m in range(0, H.M)])
        D = max([len(H.E[m]) for m in range(0, H.M)])
        count_by_size = {}
        for m in range(0, M):
            s = len(H.E[m])
            count_by_size[s] = count_by_size.get(s, 0) + 1

        print("Number of nodes: " + str(H.N))
        print("Number of hyperedges: " + str(H.M))
        print("Sum of hyperedge weights: " + str(np.sum(H.A)))
        print("Average degree of the node: " + str(float(sum_degree) / H.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / H.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Hyperedge size distribution: " + str(list(sorted(count_by_size.items()))))
        print("Connected hypergraph: " + str(H.is_connected()))
        print()

    return


def generate_hsbm(N, K, alpha_in_2, beta_out_2, lambda_div, avg_node_degree, print_info=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if N % K != 0:
        raise ValueError(f"N ({N}) must be divisible by K ({K}) for equal community sizes.")
    nodes_per_community = N // K
    community = np.repeat(np.arange(K), nodes_per_community)
    np.random.shuffle(community)

    a_2_prime = float(alpha_in_2)
    b_2_prime = float(beta_out_2)

    a_3_prime = (1 - lambda_div) * a_2_prime + lambda_div * b_2_prime
    b_3_prime = (1 - lambda_div) * b_2_prime + lambda_div * a_2_prime

    a_4_prime = (1 - lambda_div) * a_2_prime + lambda_div * (a_2_prime + b_2_prime) * 0.5
    b_4_prime = (1 - lambda_div) * b_2_prime + lambda_div * (a_2_prime + b_2_prime) * 0.5

    params_prime = {
        2: (a_2_prime, b_2_prime),
        3: (a_3_prime, b_3_prime),
        4: (a_4_prime, b_4_prime),
    }
    D = 4

    current_expected_degree = 0.0
    for s in range(2, D + 1):
        if s not in params_prime: continue
        a_s_prime, b_s_prime = params_prime[s]

        try:
            # C(nodes_per_community - 1, s - 1)
            comb_in = math.comb(nodes_per_community - 1, s - 1)
            # C(N - 1, s - 1)
            comb_total = math.comb(N - 1, s - 1)
        except ValueError:
            continue

        if comb_total == 0: continue

        in_cluster_ratio = comb_in / comb_total

        degree_contribution = a_s_prime * in_cluster_ratio + b_s_prime * (1 - in_cluster_ratio)
        current_expected_degree += degree_contribution

    if current_expected_degree <= 1e-10:
        scale_factor = 0.0
    else:
        scale_factor = avg_node_degree / current_expected_degree

    E_generated = []
    V_nodes = range(N)

    for s in range(2, D + 1):
        if s not in params_prime: continue

        a_s_prime, b_s_prime = params_prime[s]
        final_a_m = a_s_prime * scale_factor
        final_b_m = b_s_prime * scale_factor

        try:
            denominator = math.comb(N - 1, s - 1)
        except ValueError:
            continue

        if denominator == 0: continue

        p_s = min(1.0, max(0.0, final_a_m / denominator))
        q_s = min(1.0, max(0.0, final_b_m / denominator))

        for e_tuple in combinations(V_nodes, s):
            first_comm_label = community[e_tuple[0]]
            is_in_cluster = all(community[node_id] == first_comm_label for node_id in e_tuple[1:])

            prob = p_s if is_in_cluster else q_s

            if random.random() < prob:
                E_generated.append(e_tuple)

    hypergraph_M = len(E_generated)
    H = HyperGraph(N, hypergraph_M)
    H.E = list(E_generated)
    H.A = np.ones(hypergraph_M)

    U_true = np.zeros((N, K), dtype=float)
    for i in range(N):
        U_true[i, community[i]] = 1

    if print_info:
        sum_degree = sum([len(H.E[m]) for m in range(0, H.M)])
        D = max([len(H.E[m]) for m in range(0, H.M)])
        M = len(H.E)
        count_by_size = {}
        for m in range(0, M):
            s = len(H.E[m])
            count_by_size[s] = count_by_size.get(s, 0) + 1

        print("Number of nodes: " + str(H.N))
        print("Number of hyperedges: " + str(H.M))
        print("Sum of hyperedge weights: " + str(np.sum(H.A)))
        print("Average degree of the node: " + str(float(sum_degree) / H.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / H.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Hyperedge size distribution: " + str(list(sorted(count_by_size.items()))))
        print("Connected hypergraph: " + str(H.is_connected()))

    return (H, U_true)



def construct_hypergraph(N: int, M: int, E: list, A: list, Z: int = None, X: list = None, print_info: bool = True):

    node_ids = []
    for e in E:
        node_ids += list([int(v) for v in e])

    if set(node_ids) != set(range(0, N)):
        print("ERROR: node ID set is invalid.")
        return

    if len(E) != M:
        print("ERROR: hyperedge set is invalid.")
        return

    for a in A:
        if type(a) is not int or int(a) < 1:
            print("ERROR: hyperedge weight list is invalid.")
            return

    if Z is not None and X is not None:
        node_category_ids = set([int(x) for x in X])
        if node_category_ids != set(range(0, Z)):
            print("ERROR: node category ID set is invalid.")
            return

    H = HyperGraph(N, M)
    H.E = list([list(sorted(e)) for e in E])
    H.A = np.array([int(a) for a in A], dtype=int)

    if Z is not None and X is not None:
        H.Z = Z
        H.X = np.array([int(x) for x in X], dtype=int)

    if print_info:
        sum_degree = sum([len(H.E[m]) for m in range(0, H.M)])
        D = max([len(H.E[m]) for m in range(0, H.M)])
        count_by_size = {}
        for m in range(0, M):
            s = len(H.E[m])
            count_by_size[s] = count_by_size.get(s, 0) + 1

        print("Number of nodes: " + str(H.N))
        print("Number of hyperedges: " + str(H.M))
        print("Sum of hyperedge weights: " + str(np.sum(H.A)))
        print("Average degree of the node: " + str(float(sum_degree) / H.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / H.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Hyperedge size distribution: " + str(list(sorted(count_by_size.items()))))
        print("Connected hypergraph: " + str(H.is_connected()))

    return H
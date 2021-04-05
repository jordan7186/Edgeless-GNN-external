# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:43:59 2020

@author: CSE-190730
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import random
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import to_networkx, from_networkx
from random import sample, shuffle
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.model_selection import ShuffleSplit as SS
from sklearn.neighbors import NearestNeighbors
from sklearn.multiclass import OneVsRestClassifier as OVRC

def h_ratio(network, label):
    count = 0
    edgelist = sorted(network.edges)
    
    for edge in edgelist:
        if label[edge[0]] == label[edge[1]]:
            count += 1
    return count/len(edgelist)
    

def get_nmi(embedding, nodelabel):
    km = KMeans(n_clusters = max(nodelabel) + 1, random_state=0).fit_predict(embedding)
    sc = SpectralClustering(n_clusters = max(nodelabel) + 1).fit_predict(embedding)

    km_nmi = nmi(nodelabel, km)
    sc_nmi = nmi(nodelabel, sc)
    #sc_nmi = 0

    return km_nmi, sc_nmi

def get_nmi_new(val_embedding, test_embedding, train_graph, val_graph, node_label):
    num_train_nodes = len(train_graph)
    num_val_nodes = len(val_graph) - len(train_graph)
    
    #train_label = node_label[:num_train_nodes]
    val_label = node_label[num_train_nodes : num_train_nodes + num_val_nodes]
    test_label = node_label[num_train_nodes + num_val_nodes:]
    
    km_val = KMeans(n_clusters = max(node_label) + 1, random_state=0).fit_predict(val_embedding)
    #sc_val = SpectralClustering(n_clusters = max(node_label) + 1).fit_predict(val_embedding)
    
    km_test = KMeans(n_clusters = max(node_label) + 1, random_state=0).fit_predict(test_embedding)
    #sc_test = SpectralClustering(n_clusters = max(node_label) + 1).fit_predict(test_embedding)
    
    km_nmi_val = nmi(val_label, km_val[num_train_nodes:])
    #sc_nmi_val = nmi(val_label, sc_val[num_train_nodes:])
    
    km_nmi_test = nmi(test_label, km_test[num_train_nodes:])
    #sc_nmi_test = nmi(test_label, sc_test[num_train_nodes:])
    
    return km_nmi_val, 0, km_nmi_test, 0

def get_f1(embedding, nodelabel, train_graph, val_graph, test_graph, train_portion = 0.05):
    num_train_nodes = len(train_graph)
    num_val_nodes = len(val_graph) - len(train_graph)
    #num_test_nodes = len(test_graph) - len(val_graph)

    train_emb = embedding[:num_train_nodes]
    train_label = nodelabel[:num_train_nodes]
    
    splitter = SSS(n_splits = 1, train_size = train_portion)
    
    for train_index, _ in splitter.split(train_emb, train_label):
        train_emb = train_emb[train_index]
        train_label = train_label[train_index]
    
    val_emb = embedding[num_train_nodes:num_train_nodes + num_val_nodes]
    test_emb = embedding[num_train_nodes + num_val_nodes:]

    val_label = nodelabel[num_train_nodes:num_train_nodes + num_val_nodes]
    test_label = nodelabel[num_train_nodes + num_val_nodes:]

    lr = LogisticRegression(multi_class='ovr').fit(10 * train_emb, train_label)
    val_pred = lr.predict(10 * val_emb)
    test_pred = lr.predict(10 * test_emb)

    val_mic = f1_score(val_label, val_pred, average = 'micro')
    val_mac = f1_score(val_label, val_pred, average = 'macro')

    test_mic = f1_score(test_label, test_pred, average = 'micro')
    test_mac = f1_score(test_label, test_pred, average = 'macro')

    return val_mic, val_mac, test_mic, test_mac

def get_f1_new(val_embedding, test_embedding, nodelabel, train_graph, val_graph, test_graph, train_portion = 0.05):
    num_train_nodes = len(train_graph)
    num_val_nodes = len(val_graph) - len(train_graph)
    #num_test_nodes = len(test_graph) - len(val_graph)

    # Validation
    train_emb_val = val_embedding[:num_train_nodes]
    train_label = nodelabel[:num_train_nodes]
    
    splitter_val = SSS(n_splits = 1, train_size = train_portion)
    
    for train_index, _ in splitter_val.split(train_emb_val, train_label):
        train_emb_val = train_emb_val[train_index]
        train_label = train_label[train_index]
    
    val_emb = val_embedding[num_train_nodes:]
    val_label = nodelabel[num_train_nodes:num_train_nodes + num_val_nodes]
    lr_val = LogisticRegression(multi_class='ovr').fit(10 * train_emb_val, train_label)
    val_pred = lr_val.predict(10 * val_emb)
    
    # Test
    train_emb_test = test_embedding[:num_train_nodes]
    train_label = nodelabel[:num_train_nodes]
    
    splitter_test = SSS(n_splits = 1, train_size = train_portion)
    
    for train_index, _ in splitter_test.split(train_emb_test, train_label):
        train_emb_test = train_emb_test[train_index]
        train_label = train_label[train_index]
        
    test_emb = test_embedding[num_train_nodes:]
    test_label = nodelabel[num_train_nodes + num_val_nodes:]

    lr_test = LogisticRegression(multi_class='ovr').fit(10 * train_emb_test, train_label)
    test_pred = lr_test.predict(10 * test_emb)

    val_mic = f1_score(val_label, val_pred, average = 'micro')
    val_mac = f1_score(val_label, val_pred, average = 'macro')

    test_mic = f1_score(test_label, test_pred, average = 'micro')
    test_mac = f1_score(test_label, test_pred, average = 'macro')

    return val_mic, val_mac, test_mic, test_mac

def get_f1_new_multilabel(val_embedding, test_embedding, nodelabel, train_graph, val_graph, test_graph, train_portion = 0.05):
    num_train_nodes = len(train_graph)
    num_val_nodes = len(val_graph) - len(train_graph)
    #num_test_nodes = len(test_graph) - len(val_graph)

    # Validation
    train_emb_val = val_embedding[:num_train_nodes]
    train_label = nodelabel[:num_train_nodes]
    
    splitter_val = SS(n_splits = 1, train_size = train_portion)
    
    for train_index, _ in splitter_val.split(train_emb_val, train_label):
        train_emb_val = train_emb_val[train_index]
        train_label = train_label[train_index]
    
    val_emb = val_embedding[num_train_nodes:]
    val_label = nodelabel[num_train_nodes:num_train_nodes + num_val_nodes]
    lr_val = OVRC(LogisticRegression()).fit(10 * train_emb_val, train_label)
    val_pred = lr_val.predict(10 * val_emb)
    
    # Test
    train_emb_test = test_embedding[:num_train_nodes]
    train_label = nodelabel[:num_train_nodes]
    
    splitter_test = SS(n_splits = 1, train_size = train_portion)
    
    for train_index, _ in splitter_test.split(train_emb_test, train_label):
        train_emb_test = train_emb_test[train_index]
        train_label = train_label[train_index]
        
    test_emb = test_embedding[num_train_nodes:]
    test_label = nodelabel[num_train_nodes + num_val_nodes:]

    lr_test = OVRC(LogisticRegression()).fit(10 * train_emb_test, train_label)
    test_pred = lr_test.predict(10 * test_emb)

    val_mic = f1_score(val_label, val_pred, average = 'micro')
    val_mac = f1_score(val_label, val_pred, average = 'macro')

    test_mic = f1_score(test_label, test_pred, average = 'micro')
    test_mac = f1_score(test_label, test_pred, average = 'macro')

    return val_mic, val_mac, test_mic, test_mac

def get_auc(embedding, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge):
    val_label = [1] * len(pos_val_edge) + [0] * len(neg_val_edge)
    test_label = [1] * len(pos_test_edge) + [0] * len(neg_test_edge)

    val_pred = []
    test_pred = []

    val_edgelist = pos_val_edge + neg_val_edge
    test_edgelist = pos_test_edge + neg_test_edge

    for val_edge in val_edgelist:
        emb0 = embedding[val_edge[0]]
        emb1 = embedding[val_edge[1]]
        val_pred.append(expit(np.inner(emb0, emb1)))

    for test_edge in test_edgelist:
        emb0 = embedding[test_edge[0]]
        emb1 = embedding[test_edge[1]]
        test_pred.append(expit(np.inner(emb0, emb1)))

    val_auc = roc_auc_score(val_label, val_pred)
    test_auc = roc_auc_score(test_label, test_pred)

    val_ap = average_precision_score(val_label, val_pred)
    test_ap = average_precision_score(test_label, test_pred)

    return val_auc, test_auc, val_ap, test_ap

def get_auc_new(val_embedding, test_embedding, num_train, num_val, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge):
    val_label = [1] * len(pos_val_edge) + [0] * len(neg_val_edge)
    test_label = [1] * len(pos_test_edge) + [0] * len(neg_test_edge)

    val_pred = []
    test_pred = []

    val_edgelist = pos_val_edge + neg_val_edge
    test_edgelist = pos_test_edge + neg_test_edge
    
    new_test_embedding = np.insert(test_embedding, num_train, np.zeros((num_val, test_embedding.shape[1])), axis = 0)

    for val_edge in val_edgelist:
        emb0 = val_embedding[val_edge[0]]
        emb1 = val_embedding[val_edge[1]]
        val_pred.append(expit(np.inner(emb0, emb1)))

    for test_edge in test_edgelist:
        emb0 = new_test_embedding[test_edge[0]]
        emb1 = new_test_embedding[test_edge[1]]
        test_pred.append(expit(np.inner(emb0, emb1)))

    val_auc = roc_auc_score(val_label, val_pred)
    test_auc = roc_auc_score(test_label, test_pred)

    val_ap = average_precision_score(val_label, val_pred)
    test_ap = average_precision_score(test_label, test_pred)

    return val_auc, test_auc, val_ap, test_ap

def tsnePrinter(testemb, colorlist, epoch, filename, PATH):
    tsneemb = TSNE(n_components = 2, random_state = 0).fit_transform(testemb)
    plt.scatter(tsneemb[:,0], tsneemb[:,1], cmap = plt.cm.gist_rainbow, c = colorlist)
    plt.title('Training epoch: {}'.format(epoch))
    plt.savefig(PATH + '/{}/epoch_{}.jpg'.format(filename, epoch))
    plt.close()

def selective_dropout(tensor, maskindicator):
    mask = (tensor != maskindicator).int()

    num_unmasked = torch.sum(mask, axis = 1)
    tot_element = tensor.size()[-1]
    inv_p = torch.Tensor([tot_element])/ num_unmasked

    return mask * tensor * inv_p.unsqueeze(1)


def incomplete_cos_where(vector1, vector2):
    # condiser '?' as -99
    if len(vector1.shape) == 1:
        index_legit1 = set(np.where(vector1 != -99)[0])
        index_legit2 = set(np.where(vector2 != -99)[0])

        index_legit = np.array(list(index_legit1.intersection(index_legit2)))
        return cosine_similarity(vector1[index_legit].reshape(1, -1), vector2[index_legit].reshape(1, -1))[0][0]

    if len(vector1.shape) == 2:
        index_legit1 = set(np.where(vector1 != -99)[1])
        index_legit2 = set(np.where(vector2 != -99)[1])

        index_legit = np.array(list(index_legit1.intersection(index_legit2)))
        return cosine_similarity(vector1[0,index_legit].reshape(1, -1), vector2[0,index_legit].reshape(1, -1))[0][0]

def getEdgesforLP(traingraph, excludeedge):
    pos_edgelist = list(traingraph.edges)
    reverse_pos = [(edge[1], edge[0]) for edge in pos_edgelist]
    tot_poslist = pos_edgelist + reverse_pos

    excludeedge = [tuple(edge) for edge in excludeedge]
    reverse_exc = [(edge[1], edge[0]) for edge in excludeedge]
    tot_exclude = excludeedge + reverse_exc

    neg_edgelist = []

    count = 0
    num_of_samples = len(tot_poslist)

    nodes = list(traingraph)

    forbidden = tot_poslist + tot_exclude
    # while the # sampled edges are less than the desired number
    while count < num_of_samples:
        a = np.random.choice(nodes, size = 1)[0]
        b = np.random.choice(nodes, size = 1)[0]
        edge0 = (a, b)
        edge1 = (b, a)
        # Criteria: not a positive edge / not a self-loop / edge not in excludeedge
        while edge0 in forbidden or edge0 in neg_edgelist or edge1 in forbidden or edge1 in neg_edgelist or a == b:
            a = np.random.choice(nodes, size = 1)[0]
            b = np.random.choice(nodes, size = 1)[0]
            edge0 = (a, b)
            edge1 = (b, a)
        # Check for reocurrences
        neg_edgelist.append(edge0)
        neg_edgelist.append(edge1)
        count += 2

    return tot_poslist, neg_edgelist

class LinkPrediction_sigmoid():
    def __init__(self, train_posedge, train_negedge, eval_posedge, eval_negedge, embedding, mode):
        # train_posedge, train_negedge, eval_posedge, eval_negedge: list of tuples
        # embedding: numpy array
        self.embedding = embedding

        self.train_edge = train_posedge + train_negedge
        self.train_label = [1] * len(train_posedge) + [0] * len(train_negedge)

        self.eval_edge = eval_posedge + eval_negedge
        self.eval_label = [1] * len(eval_posedge) + [0] * len(eval_negedge)

        assert mode in ['sigmoid', 'logsigmoid']
        self.mode = mode

        # Train classifier in init
        self.clf = self.trainLP(self.train_edge, self.train_label, self.embedding, self.mode)

    def trainLP(self, train_edge, train_label, embedding, mode):
        trainemb = []

        for edge in train_edge:
            emb0 = embedding[edge[0]]
            emb1 = embedding[edge[1]]
            if mode == 'sigmoid':
                trainemb.append(expit(np.inner(emb0, emb1)))
            else:
                trainemb.append(np.log(expit(np.inner(emb0, emb1))))

        trainemb = np.array(trainemb)

        self.trainemb = trainemb

        clf = LogisticRegression(random_state=0).fit(trainemb.reshape(-1, 1), train_label)

        return clf

def node_split(graph, feature, nodelabel, val_portion, test_portion, lcc = True):
    if lcc:
        # Split test/train by random sampling
        num_test_node = int(len(graph) * test_portion)
        num_val_node = int(len(graph) * val_portion)
        num_train_node = len(graph) - num_val_node - num_test_node

        train_graph = random_sample_vanilla(graph, num_train_node)
        # Relabel node index: [train, test]
        train_node = sorted(train_graph)
        remain_node = list(set(graph) - set (train_graph))
        shuffle(remain_node)
        val_node = sorted(remain_node[:num_val_node])
        test_node = sorted(remain_node[num_val_node:])

        mapping = dict(zip(train_node + val_node + test_node, sorted(graph)))
        total_index = np.array(train_node + val_node + test_node)
        new_graph = relabel(graph, mapping)

        new_train_graph = new_graph.subgraph(list(range(len(train_node))))
        new_val_graph = new_graph.subgraph(list(range(len(train_node + val_node))))

        pos_val_edge = list(set(new_val_graph.edges) - set(new_train_graph.edges))
        pos_test_edge = list(set(new_graph.edges) - set(new_val_graph.edges))

        new_train_node = list(range(len(new_train_graph)))
        new_val_node = list(range(len(new_train_graph), len(new_val_graph)))
        new_test_node = list(range(len(new_val_graph), len(new_graph)))

        neg_val_edge = []
        neg_test_edge = []

        for _ in range(len(pos_val_edge)):
            new_edge = (sample(new_train_node, 1)[0], sample(new_val_node, 1)[0])
            while new_edge in new_val_graph:
                new_edge = (sample(new_train_node, 1)[0], sample(new_test_node, 1)[0])
            neg_val_edge.append(new_edge)

        for _ in range(len(pos_test_edge)):
            new_edge = (sample(new_train_node + new_val_node, 1)[0], sample(new_test_node, 1)[0])
            while new_edge in new_graph:
                new_edge = (sample(new_train_node + new_val_node, 1)[0], sample(new_test_node, 1)[0])
            neg_test_edge.append(new_edge)

        val_package = (pos_val_edge, neg_val_edge)
        test_package = (pos_test_edge, neg_test_edge)
        graph_package = (new_train_graph, new_val_graph, new_graph)

        return graph_package, total_index, feature[total_index], nodelabel[total_index], val_package, test_package

    else:
        # Split test/train by random sampling
        num_test_node = int(len(graph) * test_portion)
        num_val_node = int(len(graph) * val_portion)
        num_train_node = len(graph) - num_val_node - num_test_node

        train_node = sorted(sample(list(graph), num_train_node))
        remain_node = sorted(list(set(graph) - set(train_node)))
        val_node = sample(remain_node, num_val_node)
        test_node = sorted(list(set(remain_node) - set(val_node)))

        # Relabel node index: [train, test]
        mapping = dict(zip(train_node + val_node + test_node, sorted(graph)))
        total_index = np.array(train_node + val_node + test_node)
        new_graph = relabel(graph, mapping)

        new_train_graph = new_graph.subgraph(list(range(len(train_node))))
        new_val_graph = new_graph.subgraph(list(range(len(train_node + val_node))))

        pos_val_edge = list(set(new_val_graph.edges) - set(new_train_graph.edges))
        pos_test_edge = list(set(new_graph.edges) - set(new_val_graph.edges))

        new_train_node = list(range(len(new_train_graph)))
        new_val_node = list(range(len(new_train_graph), len(new_val_graph)))
        new_test_node = list(range(len(new_val_graph), len(new_graph)))

        neg_val_edge = []
        neg_test_edge = []

        for _ in range(len(pos_val_edge)):
            new_edge = (sample(new_train_node, 1)[0], sample(new_val_node, 1)[0])
            while new_edge in new_val_graph:
                new_edge = (sample(new_train_node, 1)[0], sample(new_test_node, 1)[0])
            neg_val_edge.append(new_edge)

        for _ in range(len(pos_test_edge)):
            new_edge = (sample(new_train_node + new_val_node, 1)[0], sample(new_test_node, 1)[0])
            while new_edge in new_graph:
                new_edge = (sample(new_train_node + new_val_node, 1)[0], sample(new_test_node, 1)[0])
            neg_test_edge.append(new_edge)

        val_package = (pos_val_edge, neg_val_edge)
        test_package = (pos_test_edge, neg_test_edge)
        graph_package = (new_train_graph, new_val_graph, new_graph)

        return graph_package, total_index, feature[total_index], nodelabel[total_index], val_package, test_package

def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]

    # Self-loops
    A_loop = sp.eye(N) + A

    # Symmetric transition matrix
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec

    return T_S

def model_generator(model_name, setting):
    assert model_name in ['SGC', 'GCN', 'GraphSAGE','GAE', 'VGAE']

    if model_name == 'SGC':
        pass
    elif model_name == 'GCN':
        pass
    elif model_name == 'GraphSAGE':
        pass
    elif model_name == 'GAE':
        pass
    elif model_name == 'VGAE':
        pass

def first_order_loss_sig(embed1, embed2, weight = None):
    if weight == None:
        mulbatch = torch.mul(embed1, embed2)
        return -torch.mean(F.logsigmoid(torch.sum(mulbatch, dim = 0)))
    else:
        mulbatch = torch.mul(embed1, embed2)
        return -torch.mean(weight * F.logsigmoid(torch.sum(mulbatch, dim = 0)))

def negative_loss_sig(embed1, embed2):
    negbatch = -torch.mul(embed1.view(len(embed1), 1, -1), embed2)
    return -torch.mean(torch.sum(F.logsigmoid(torch.sum(negbatch, dim = 2)), dim = 1))

def relabel(G, mapping):
    H = nx.Graph()
    old = list(mapping.keys())
    H.add_nodes_from([mapping[oldnode] for oldnode in old])
    H.add_edges_from([(mapping[oldedge[0]], mapping[oldedge[1]]) for oldedge in G.edges])
    # Add this for correct drawing
    H._node = sorted(H._node)
    return H

def get_class_index(nodelabel_raw, class_list):
    filter_list = np.zeros(len(nodelabel_raw), dtype = bool)
    for index, class_num in enumerate(nodelabel_raw):
        if class_num in class_list:
            filter_list[index] = True
        else:
            filter_list[index] = False
    
    return filter_list

def preprocess(data, lcc = True, is_sorted = True, class_sample = False):
    raw_graph = to_networkx(data, to_undirected = True)
    raw_label = data.y.numpy()
    raw_feature = data.x.numpy()
    
    if class_sample != False:
        print("class sample")
        filter_data = get_class_index(raw_label, class_sample)
        raw_label = raw_label[filter_data]
        raw_feature = raw_feature[filter_data]
        raw_adj = nx.to_numpy_array(raw_graph)
        reduced_adj = raw_adj[np.ix_(np.arange(len(raw_graph))[filter_data], np.arange(len(raw_graph))[filter_data])]
        raw_graph = nx.to_networkx_graph(reduced_adj)

    if is_sorted == False:
        mapping = dict(zip(raw_graph, list(range(len(raw_graph)))))
        raw_graph = relabel(raw_graph, mapping)

    raw_nodelist = np.array(raw_graph)

    if lcc:
        connected_comp = sorted(nx.connected_components(raw_graph), key=len, reverse=True)
        # total_set = set(raw_nodelist)
        lcc_set = connected_comp[0]
        # rest_set = total_set - lcc_set
        lcc_graph = raw_graph.subgraph(sorted(lcc_set))

        mapping = dict(zip(sorted(lcc_graph), list(range(len(lcc_graph)))))
        lcc_new = relabel(lcc_graph, mapping)
        feature_new = raw_feature[np.array(lcc_graph)]
        label_new = raw_label[np.array(lcc_graph)]

        return lcc_new, feature_new, label_new
    else:
        return raw_graph, raw_feature, raw_label

def preprocess_sample(raw_graph, raw_feature, raw_label, lcc = True, is_sorted = True, class_sample = False):
    if class_sample != False:
        print("class sample")
        filter_data = get_class_index(raw_label, class_sample)
        raw_label = raw_label[filter_data]
        raw_feature = raw_feature[filter_data]
        raw_adj = nx.to_numpy_array(raw_graph)
        reduced_adj = raw_adj[np.ix_(np.arange(len(raw_graph))[filter_data], np.arange(len(raw_graph))[filter_data])]
        raw_graph = nx.to_networkx_graph(reduced_adj)

    if is_sorted == False:
        mapping = dict(zip(raw_graph, list(range(len(raw_graph)))))
        raw_graph = relabel(raw_graph, mapping)

    raw_nodelist = np.array(raw_graph)

    if lcc:
        connected_comp = sorted(nx.connected_components(raw_graph), key=len, reverse=True)
        # total_set = set(raw_nodelist)
        lcc_set = connected_comp[0]
        # rest_set = total_set - lcc_set
        lcc_graph = raw_graph.subgraph(sorted(lcc_set))

        mapping = dict(zip(sorted(lcc_graph), list(range(len(lcc_graph)))))
        lcc_new = relabel(lcc_graph, mapping)
        feature_new = raw_feature[np.array(lcc_graph)]
        label_new = raw_label[np.array(lcc_graph)]

        return lcc_new, feature_new, label_new
    else:
        return raw_graph, raw_feature, raw_label
    
def random_sample_vanilla(graph, sample_portion):
    sample_graph = deepcopy(graph)
    if type(sample_portion) == float:
        sample_size = int(len(graph) * sample_portion)
    elif type(sample_portion) == int:
        sample_size = sample_portion
    else:
        raise ValueError("sample_portion must be either int > 0 or 0 < float <= 1")

    print("Sampling nodes... sample {} out of {} nodes.".format(sample_size, len(graph)))

    nodeset = set(sample_graph)

    while len(nodeset) > sample_size:
        delete_node = random.sample(list(nodeset), 1)[0]
        nodeset.remove(delete_node)

        test_graph = sample_graph.subgraph(nodeset)

        if nx.is_connected(test_graph):
            sample_graph = sample_graph.subgraph(nodeset)
        else:
            nodeset.add(delete_node)

    return sample_graph

def test_node_split(graph, feature, nodelabel, test_portion):
    # Split test/train by random sampling
    num_test_node = int(len(graph) * test_portion)
    num_train_node = len(graph) - num_test_node

    train_graph = random_sample_vanilla(graph, num_train_node)
    # Relabel node index: [train, test]
    train_node = sorted(train_graph)
    test_node = sorted(list(set(graph) - set (train_graph)))
    mapping = dict(zip(train_node + test_node, sorted(graph)))
    total_index = np.array(train_node + test_node)
    new_graph = relabel(graph, mapping)
    # True train graph
    new_train_graph = new_graph.subgraph(list(range(len(train_node))))

    test_pos_edge = list(set(new_graph.edges) - set(new_train_graph.edges))

    new_train_node = list(range(len(new_train_graph)))
    new_test_node = list(range(len(new_train_graph), len(new_graph)))

    test_neg_edge = []

    for _ in range(len(test_pos_edge)):
        new_edge = (random.sample(new_train_node, 1)[0], random.sample(new_test_node, 1)[0])
        while new_edge in new_graph:
            new_edge = (random.sample(new_train_node, 1)[0], random.sample(new_test_node, 1)[0])
        test_neg_edge.append(new_edge)

    return new_graph, total_index, feature[total_index], nodelabel[total_index], test_pos_edge, test_neg_edge

def constkNNGraph_new(feature_ordered, train_node_num, val_node_num, test_node_num, k, algorithm = 'brute'):
    train_feature = feature_ordered[:train_node_num]
    val_feature = feature_ordered[train_node_num:train_node_num+val_node_num]
    test_feature = feature_ordered[train_node_num+val_node_num:]
    
    val_input_feature = np.concatenate((train_feature, val_feature), axis = 0)
    test_input_feature = np.concatenate((train_feature, test_feature), axis = 0)
    
    adj_val = np.zeros((train_node_num + val_node_num, train_node_num + val_node_num))
    adj_test = np.zeros((train_node_num + test_node_num, train_node_num + test_node_num))
    
    # Normalized features + Euclidean  = cosine
    # https://stackoverflow.com/questions/34144632/using-cosine-distance-with-scikit-learn-kneighborsclassifier/34145444
    nbrs = NearestNeighbors(n_neighbors = k, algorithm = algorithm).fit(train_feature)
    adj_kadj_train = nbrs.kneighbors_graph(train_feature).toarray() - np.eye(train_node_num)
    kadj_train_graph = nx.to_networkx_graph(adj_kadj_train)
    adj_val[:train_node_num,:train_node_num] = adj_kadj_train
    adj_test[:train_node_num,:train_node_num] = adj_kadj_train
    
    val_index_neigh = nbrs.kneighbors(val_feature, return_distance = False)
    val_index_neigh = np.squeeze(val_index_neigh.reshape(1,-1))
    val_index_self = np.repeat(np.arange(train_node_num, train_node_num + val_node_num, dtype = int), k)
    adj_val[val_index_self,val_index_neigh] = 1
    adj_val[val_index_neigh,val_index_self] = 1
    kadj_val_graph = nx.to_networkx_graph(adj_val)
    
    test_index_neigh = nbrs.kneighbors(test_feature, return_distance = False)
    test_index_neigh = np.squeeze(test_index_neigh.reshape(1,-1))
    test_index_self = np.repeat(np.arange(train_node_num, train_node_num + test_node_num, dtype = int), k)
    adj_test[test_index_self,test_index_neigh] = 1
    adj_test[test_index_neigh,test_index_self] = 1
    kadj_test_graph = nx.to_networkx_graph(adj_test)
    
    return [kadj_train_graph, kadj_val_graph, kadj_test_graph], [train_feature, val_input_feature, test_input_feature]

def constkNNGraph_random(feature_ordered, train_node_num, val_node_num, test_node_num, k, algorithm = 'brute', train = 'knn'):
    train_feature = feature_ordered[:train_node_num]
    val_feature = feature_ordered[train_node_num:train_node_num+val_node_num]
    test_feature = feature_ordered[train_node_num+val_node_num:]
    
    val_input_feature = np.concatenate((train_feature, val_feature), axis = 0)
    test_input_feature = np.concatenate((train_feature, test_feature), axis = 0)
    
    adj_val = np.zeros((train_node_num + val_node_num, train_node_num + val_node_num))
    adj_test = np.zeros((train_node_num + test_node_num, train_node_num + test_node_num))

    if train == 'knn':
        nbrs = NearestNeighbors(n_neighbors = k, algorithm = algorithm).fit(train_feature)
        adj_kadj_train = nbrs.kneighbors_graph(train_feature).toarray() - np.eye(train_node_num)
        kadj_train_graph = nx.to_networkx_graph(adj_kadj_train)
    elif train == 'random':
        kadj_train_graph = nx.barabasi_albert_graph(len(train_feature), 2)
        adj_kadj_train = nx.to_numpy_array(kadj_train_graph)
    adj_val[:train_node_num,:train_node_num] = adj_kadj_train
    adj_test[:train_node_num,:train_node_num] = adj_kadj_train
    
    val_index_neigh = np.random.choice(np.arange(train_node_num), size = (val_node_num, k))
    val_index_neigh = np.squeeze(val_index_neigh.reshape(1,-1))
    val_index_self = np.repeat(np.arange(train_node_num, train_node_num + val_node_num, dtype = int), k)
    adj_val[val_index_self,val_index_neigh] = 1
    adj_val[val_index_neigh,val_index_self] = 1
    kadj_val_graph = nx.to_networkx_graph(adj_val)
    
    test_index_neigh = np.random.choice(np.arange(train_node_num), size = (test_node_num, k))
    test_index_neigh = np.squeeze(test_index_neigh.reshape(1,-1))
    test_index_self = np.repeat(np.arange(train_node_num, train_node_num + test_node_num, dtype = int), k)
    adj_test[test_index_self,test_index_neigh] = 1
    adj_test[test_index_neigh,test_index_self] = 1
    kadj_test_graph = nx.to_networkx_graph(adj_test)
    
    return [kadj_train_graph, kadj_val_graph, kadj_test_graph], [train_feature, val_input_feature, test_input_feature]

def constkNNGraph(distmatrix, algorithm, k):
    """
    Input

    distmatrix: a N X N adjacency matrix where the elements
    are the cosine similarity between feature vectors.

    algorithm: a string, either 'knn' or 'mknn' or 'cknn'.
    If algorithm is 'cknn', delta is 1 by default.

    k: hyperparameter for knn methods.

    Output

    featuregraphknn: nx.Graph() containing kNN graph from feature. Not re-indexed.
    """
    assert algorithm in ['knn', 'mknn', 'cknn']

    delta = 1
    N = distmatrix.shape[0]
    adjmatrix = np.zeros((N,N))

    if algorithm == 'knn' or algorithm == 'mknn':
        for index, row in enumerate(distmatrix):
            # Get indices for top k values
            # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
            ind = np.argpartition(row, -k)[-k:]
            adjmatrix[index][ind] = 1
    elif algorithm == 'cknn':
        threshold = []
        for index, row in enumerate(distmatrix):
            ind = np.argpartition(row, -k)[-k:]
            threshold.append(min(distmatrix[index][ind]))

    if algorithm == 'knn':
        adjmatrixknn = np.maximum(adjmatrix, adjmatrix.T)

        featuregraph = nx.from_numpy_array(adjmatrixknn)
    elif algorithm == 'mknn':
        adjmatrixmknn = np.minimum(adjmatrix, adjmatrix.T)

        featuregraph = nx.from_numpy_array(adjmatrixmknn)
    elif algorithm == 'cknn':
        adjmatrixcknn = np.zeros((N,N))
        for i in range(N):
            threshold1 = threshold[i]
            for j in range(i+1, N):
                threshold2 = threshold[j]
                cut = delta * np.sqrt(threshold1 * threshold2)
                if distmatrix[i][j] > cut:
                    adjmatrixcknn[i][j] = 1
                    adjmatrixcknn[j][i] = 1

        featuregraph = nx.from_numpy_array(adjmatrixcknn)

    return featuregraph

def constDistMatrix(sourcefeatures, is_incomplete = False):
    """
    Input

    sourcefeatures: sampled feature matrix. Not re-indexed.

    Output

    distmatrix: a N X N adjacency matrix where the elements
    are the cosine similarity between feature vectors.
    """
    N = sourcefeatures.shape[0]
    distmatrix = np.zeros((N,N))

    start = time.time()
    istart = time.time()
    for i in range(N):
        for j in range(i+1, N):
            if is_incomplete:
                dist = incomplete_cos_where(sourcefeatures[i], sourcefeatures[j])
            else:
                dist = cosine_similarity(sourcefeatures[i].reshape(1, -1), sourcefeatures[j].reshape(1, -1))
            distmatrix[i][j] = dist
            distmatrix[j][i] = dist
        if i % 100 == 0 and i > 0:
            print('%d ~ %d computation time: %.2f sec' % (i - 100, i, time.time() - istart))
            istart = time.time()
    print('Total time: %.2f sec' % (time.time() - start))

    return distmatrix

def addTestNode_fast(traingraph, trainfeatures, testfeatures, k, is_incomplete = False):
    """
    Input

    traingraph: kNN graph of training nodes.

    trainfeatures: feature matrix of training nodes.

    testfeatures: feature matrix of test nodes.

    k: parameter for kNN

    Output

    testgraph: kNN graph built upon traingraph where the test node is added.
    """
    trainadj = nx.to_numpy_matrix(traingraph)

    tempfeatures = np.concatenate((trainfeatures, testfeatures))

    numtrain = trainfeatures.shape[0]
    totnum = tempfeatures.shape[0]

    adjmatrix_dist = np.zeros((totnum, totnum))
    adjmatrix = np.zeros((totnum, totnum))
    adjmatrix[0:numtrain, 0:numtrain] = trainadj

    for row in range(numtrain, totnum):
        for column in range(row):
            if is_incomplete:
                dist = incomplete_cos_where(tempfeatures[row], tempfeatures[column])
            else:
                dist = cosine_similarity(tempfeatures[row].reshape(1, -1), tempfeatures[column].reshape(1, -1))
            adjmatrix_dist[row, column] = dist
            adjmatrix_dist[column, row] = dist
    print("Calculating similarity for test node complete")

    for index in range(numtrain, totnum):
        row = adjmatrix_dist[index]
        # Get indices for top k values
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        ind = np.argpartition(row, -k)[-k:]
        adjmatrix[index][ind] = 1

    print("Construct fast kNN graph complete")
    adjmatrixknn = np.maximum(adjmatrix, adjmatrix.T)
    testgraph = nx.from_numpy_array(adjmatrixknn)

    return testgraph

def addTestNode_fast_v2(traingraph, distmatrix, train_index, test_index, k, is_incomplete = False):
    """
    Input

    traingraph: kNN graph of training nodes.

    trainfeatures: feature matrix of training nodes.

    testfeatures: feature matrix of test nodes.

    k: parameter for kNN

    Output

    testgraph: kNN graph built upon traingraph where the test node is added.
    """
    trainadj = nx.to_numpy_matrix(traingraph)

    totindex = np.concatenate((train_index, test_index))

    numtrain = len(train_index)
    totnum = len(train_index) + len(test_index)

    adjmatrix_dist = distmatrix[np.ix_(totindex, totindex)]
    adjmatrix = np.zeros((totnum, totnum))
    adjmatrix[0:numtrain, 0:numtrain] = trainadj

    print("Calculating similarity for test node complete")

    for index in range(numtrain, totnum):
        row = adjmatrix_dist[index]
        # Get indices for top k values
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        ind = np.argpartition(row, -k)[-k:]
        adjmatrix[index][ind] = 1

    print("Construct fast kNN graph complete")
    adjmatrixknn = np.maximum(adjmatrix, adjmatrix.T)
    testgraph = nx.from_numpy_array(adjmatrixknn)

    return testgraph


def genTestData(test_knn, test_feature, test_pos_edge, test_neg_edge, train_num):
    traindata = from_networkx(test_knn)
    traindata.x = torch.Tensor(test_feature)
    traindata.test_pos_edge = torch.Tensor(test_pos_edge).long().t()
    traindata.test_neg_edge = torch.Tensor(test_neg_edge).long().t()
    traindata.train_mask = torch.BoolTensor([True] * train_num + [False] * (len(test_knn) - train_num))

    return traindata

def genTrainData(train_knn, train_feature):
    traindata = from_networkx(train_knn)
    traindata.x = torch.Tensor(train_feature)

    return traindata


def pos_loss(embed1, embed2, weight = None):
    if weight == None:
        mulbatch = torch.mul(embed1, embed2)
        return -torch.mean(F.logsigmoid(torch.sum(mulbatch, dim = 0)))
    else:
        mulbatch = torch.mul(embed1, embed2)
        return -torch.mean(weight * F.logsigmoid(torch.sum(mulbatch, dim = 0)))

def neg_loss(embed1, embed2):
    negbatch = -torch.mul(embed1.view(len(embed1), 1, -1), embed2)
    return -torch.mean(torch.sum(F.logsigmoid(torch.sum(negbatch, dim = 2)), dim = 1))

def get_link_labels(pos_edge_index, neg_edge_index, device):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def get_hops(G, K):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.
    Parameters
    ----------
    G : nx.Graph
        The graph represented as a NetworkX graph.
    K : int
        The maximum hopness to consider.
    Returns
    -------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as NetworkX graphs.
    """
    A = nx.to_scipy_sparse_matrix(G)

    hops = {1: A.tolil()}
    hops[1].setdiag(0)

    for h in range(2, K + 1):
        # compute the next ring
        next_hop = hops[h - 1].dot(A)
        next_hop[next_hop > 0] = 1

        # make sure that we exclude visited n/edges
        for prev_h in range(1, h):
            next_hop -= next_hop.multiply(hops[prev_h])

        next_hop = next_hop.tolil()
        next_hop.setdiag(0)

        hops[h] = next_hop

    for h in range(1, K + 1):
        curr_matrix = hops[h]
        hops[h] = nx.from_scipy_sparse_matrix(curr_matrix)

    return hops

class struct_loader(torch.utils.data.Dataset):
    def __init__(self, G, negative_ratio, weight = False):
        print("Preparing to feed edges")
        self.G = G
        self.negative_ratio = negative_ratio
        self.table = np.array([G.degree[node] for node in G]) ** 0.75
        self.weight = weight

        pos0 = [(int(data.split()[0]), int(data.split()[1])) for data in list(nx.generate_edgelist(G, data=False)) if int(data.split()[0]) != int(data.split()[1])]
        pos1 = [(int(data.split()[1]), int(data.split()[0])) for data in list(nx.generate_edgelist(G, data=False)) if int(data.split()[0]) != int(data.split()[1])]
        pos_edge = pos0 + pos1
        pos_edge = np.array(pos_edge)

        self.target = np.array(pos_edge[:,0])
        self.context = np.array(pos_edge[:,1])

        if weight is not False:
            weight_list = []
            weight_list = [float(data.split()[-1]) for data in list(nx.generate_edgelist(G, data=['weight']))] * 2
            weight_list = np.array(weight_list)
            self.target_weight = weight_list[:,0].tolist()
            self.context_weight = weight_list[:,1].tolist()

        print("Preparing for Negative sampling")
        neg_table = [self.table[np.array(list(nx.non_neighbors(self.G, node)))] for node in G]
        self.neg_table = np.array([neg_array / sum(neg_array) for neg_array in neg_table])
        self.non_neigh = np.array([np.array(list(nx.non_neighbors(self.G, node))) for node in G])

    def __getitem__(self, index):
        index = np.array(index)
        curr_target = self.target[index]

        neg_samples = np.random.choice(self.non_neigh[curr_target], size = self.negative_ratio, p = self.neg_table[curr_target]).tolist()

        if self.weight is False:
            return torch.LongTensor([self.target[index], self.context[index]] + neg_samples)
        else:
            return torch.LongTensor([self.target[index], self.context[index]] + neg_samples + [self.target_weight[index] + self.context_weight[index]])

    def __len__(self):
        return len(self.target)

class struct_loader_vanilla(torch.utils.data.Dataset):
    def __init__(self, G, negative_ratio):
        print("Preparing to feed edges")
        self.G = G
        self.negative_ratio = negative_ratio
        self.table = np.array([G.degree[node] for node in G]) ** 0.75

        pos0 = [(int(data.split()[0]), int(data.split()[1])) for data in list(nx.generate_edgelist(G, data=False)) if int(data.split()[0]) != int(data.split()[1])]
        pos1 = [(int(data.split()[1]), int(data.split()[0])) for data in list(nx.generate_edgelist(G, data=False)) if int(data.split()[0]) != int(data.split()[1])]
        pos_edge = pos0 + pos1
        pos_edge = np.array(pos_edge)

        self.target = np.array(pos_edge[:,0])
        self.context = np.array(pos_edge[:,1])

        print("Not negative sampling")
        self.non_neigh = np.array([np.array(list(nx.non_neighbors(self.G, node))) for node in G])

    def __getitem__(self, index):
        index = np.array(index)
        curr_target = self.target[index]

        neg_samples = np.random.choice(self.non_neigh[curr_target], size = self.negative_ratio).tolist()

        return torch.LongTensor([self.target[index], self.context[index]] + neg_samples)

    def __len__(self):
        return len(self.target)

class struct_loader_ver1(torch.utils.data.Dataset):
    def __init__(self, G, hops, k):
        print("Initialization...")
        self.G = G
        self.hops = hops
        self.k = k
        assert k > 1
        #self.table = np.array([G.degree[node] for node in G]) ** 0.75

        pos0 = [(int(data.split()[0]), int(data.split()[1])) for data in list(nx.generate_edgelist(G, data=False)) if int(data.split()[0]) != int(data.split()[1])]
        pos1 = [(int(data.split()[1]), int(data.split()[0])) for data in list(nx.generate_edgelist(G, data=False)) if int(data.split()[0]) != int(data.split()[1])]
        pos_edge = pos0 + pos1
        pos_edge = np.array(pos_edge)

        self.target = np.array(pos_edge[:,0])
        self.context = np.array(pos_edge[:,1])

        print("Preparing for higher order neighbors")
        self.neigh_table = {}

        for hop in range(2, self.k + 1):
            self.neigh_table[hop] = [np.array(list(nx.neighbors(hops[hop], node))) for node in G]

    def __getitem__(self, index):
        index = np.array(index)
        curr_target = self.target[index]
        curr_context = self.context[index]

        sample = [curr_target, curr_context]

        for hop in range(2, self.k + 1):
            pick = np.random.choice(self.neigh_table[hop][curr_target], size = 1)[0]
            sample.append(pick)

            if hop == 2:
                coeff = list(nx.jaccard_coefficient(self.G, [(curr_target, pick)]))[0][-1]

        return [torch.LongTensor(sample), coeff]

    def __len__(self):
        return len(self.target)

class struct_loader_ver2(torch.utils.data.Dataset):
    def __init__(self, G, negative_ratio, twohop, is_NS = True):
        print("Preparing to feed edges")
        self.G = G
        self.negative_ratio = negative_ratio
        if is_NS:
            self.table = np.array([G.degree[node] for node in G]) ** 0.75
        else:
            self.table = np.array([1 for node in G])

        pos0 = [(int(data.split()[0]), int(data.split()[1])) for data in list(nx.generate_edgelist(G, data=False)) if int(data.split()[0]) != int(data.split()[1])]
        pos1 = [(int(data.split()[1]), int(data.split()[0])) for data in list(nx.generate_edgelist(G, data=False)) if int(data.split()[0]) != int(data.split()[1])]
        pos_edge = pos0 + pos1
        pos_edge = np.array(pos_edge)

        self.target = np.array(pos_edge[:,0])
        self.context = np.array(pos_edge[:,1])

        print("Preparing for Negative sampling")
        neg_table = [self.table[np.array(list(nx.non_neighbors(self.G, node)))] for node in G]
        self.neg_table = np.array([neg_array / sum(neg_array) for neg_array in neg_table])
        self.non_neigh = np.array([np.array(list(nx.non_neighbors(self.G, node))) for node in G])

        print("Preparing for second hop neighbors")
        self.twohop_table = [np.array(list(nx.neighbors(twohop[2], node))) for node in G]

    def __getitem__(self, index):
        index = np.array(index)
        curr_target = self.target[index]

        neg_samples = np.random.choice(self.non_neigh[curr_target], size = self.negative_ratio, p = self.neg_table[curr_target]).tolist()
        twohop_sample = np.random.choice(self.twohop_table[curr_target], size = 1).tolist()
        coeff = [list(nx.jaccard_coefficient(self.G, [(curr_target, twohop_sample[0])]))[0][-1]]

        return [torch.LongTensor([self.target[index], self.context[index]] + neg_samples + twohop_sample), torch.Tensor(coeff)]

    def __len__(self):
        return len(self.target)

##############################################################################
# Graph2Gauss
##############################################################################
import scipy.sparse as sp
import warnings
import itertools

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import normalize


def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.
    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)
    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix
    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    p_val : float
        Percentage of validation edges. Default p_val=0.10
    p_test : float
        Percentage of test edges. Default p_test=0.05
    seed : int
        Seed for numpy.random. Default seed=0
    neg_mul : int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    every_node : bool
        Make sure each node appears at least once in the train set. Default every_node=True
    connected : bool
        Make sure the training graph is still connected after the split
    undirected : bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    set_ops : bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    asserts : bool
        Unit test like checks. Default asserts=False
    Returns
    -------
    train_ones : array-like, shape [n_train, 2]
        Indices of the train edges
    val_ones : array-like, shape [n_val, 2]
        Indices of the validation edges
    val_zeros : array-like, shape [n_val, 2]
        Indices of the validation non-edges
    test_ones : array-like, shape [n_test, 2]
        Indices of the test edges
    test_zeros : array-like, shape [n_test, 2]
        Indices of the test non-edges
    """
    assert p_val + p_test > 0
    assert A.max() == 1  # no weights
    assert A.min() == 0  # no negative edges
    assert A.diagonal().sum() == 0  # no self-loops
    assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  # consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    E = A.nnz
    N = A.shape[0]
    s_train = int(E * (1 - p_val - p_test))

    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            assert sp.csgraph.connected_components(A)[0] == 1  # make sure original graph is connected
            A_hold = sp.csgraph.minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = edge_cover(A)

                # make sure the training percentage is not smaller than len(edge_cover)/E when every_node is set to True
                min_size = hold_edges.shape[0]
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))
            else:
                # make sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        # discard ones
        random_sample = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        # discard duplicates
        random_sample = random_sample[np.unique(random_sample[:, 0] * N + random_sample[:, 1], return_index=True)[1]]
        # only take as much as needed
        test_zeros = np.row_stack(random_sample)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or sp.csgraph.connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    return train_ones, val_ones, val_zeros, test_ones, test_zeros


def sparse_feeder(M):
    """
    Prepares the input matrix into a format that is easy to feed into tensorflow's SparseTensor
    Parameters
    ----------
    M : scipy.sparse.spmatrix
        Matrix to be fed
    Returns
    -------
    indices : array-like, shape [n_edges, 2]
        Indices of the sparse elements
    values : array-like, shape [n_edges]
        Values of the sparse elements
    shape : array-like
        Shape of the matrix
    """
    M = sp.coo_matrix(M)
    return np.vstack((M.row, M.col)).T, M.data, M.shape


def cartesian_product(x, y):
    """
    Form the cartesian product (i.e. all pairs of values) between two arrays.
    Parameters
    ----------
    x : array-like, shape [Nx]
        Left array in the cartesian product
    y : array-like, shape [Ny]
        Right array in the cartesian product
    Returns
    -------
    xy : array-like, shape [Nx * Ny]
        Cartesian product
    """
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


def score_link_prediction(labels, scores):
    """
    Calculates the area under the ROC curve and the average precision score.
    Parameters
    ----------
    labels : array-like, shape [N]
        The ground truth labels
    scores : array-like, shape [N]
        The (unnormalized) scores of how likely are the instances
    Returns
    -------
    roc_auc : float
        Area under the ROC curve score
    ap : float
        Average precision score
    """

    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def score_node_classification(features, z, p_labeled=0.1, n_repeat=10, norm=False):
    """
    Train a classifier using the node embeddings as features and reports the performance.
    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    z : array-like, shape [N]
        The ground truth labels
    p_labeled : float
        Percentage of nodes to use for training the classifier
    n_repeat : int
        Number of times to repeat the experiment
    norm
    Returns
    -------
    f1_micro: float
        F_1 Score (micro) averaged of n_repeat trials.
    f1_micro : float
        F_1 Score (macro) averaged of n_repeat trials.
    """
    lrcv = LogisticRegressionCV()

    if norm:
        features = normalize(features)

    trace = []
    for seed in range(n_repeat):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        lrcv.fit(features[split_train], z[split_train])
        predicted = lrcv.predict(features[split_test])

        f1_micro = f1_score(z[split_test], predicted, average='micro')
        f1_macro = f1_score(z[split_test], predicted, average='macro')

        trace.append((f1_micro, f1_macro))

    return np.array(trace).mean(0)


def get_hops_g2g(A, K):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The graph represented as a sparse matrix
    K : int
        The maximum hopness to consider.
    Returns
    -------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    """
    hops = {1: A.tolil(), -1: A.tolil()}
    hops[1].setdiag(0)

    for h in range(2, K + 1):
        # compute the next ring
        next_hop = hops[h - 1].dot(A)
        next_hop[next_hop > 0] = 1

        # make sure that we exclude visited n/edges
        for prev_h in range(1, h):
            next_hop -= next_hop.multiply(hops[prev_h])

        next_hop = next_hop.tolil()
        next_hop.setdiag(0)

        hops[h] = next_hop
        hops[-1] += next_hop

    return hops


def sample_last_hop(A, nodes):
    """
    For each node in nodes samples a single node from their last (K-th) neighborhood.
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix encoding which nodes belong to any of the 1, 2, ..., K-1, neighborhoods of every node
    nodes : array-like, shape [N]
        The nodes to consider
    Returns
    -------
    sampled_nodes : array-like, shape [N]
        The sampled nodes.
    """
    N = A.shape[0]

    sampled = np.random.randint(0, N, len(nodes))

    nnz = A[nodes, sampled].nonzero()[1]
    while len(nnz) != 0:
        new_sample = np.random.randint(0, N, len(nnz))
        sampled[nnz] = new_sample
        nnz = A[nnz, new_sample].nonzero()[1]

    return sampled


def sample_all_hops(hops, nodes=None):
    """
    For each node in nodes samples a single node from all of their neighborhoods.
    Parameters
    ----------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    nodes : array-like, shape [N]
        The nodes to consider
    Returns
    -------
    sampled_nodes : array-like, shape [N, K]
        The sampled nodes.
    """

    N = hops[1].shape[0]

    if nodes is None:
        nodes = np.arange(N)

    return np.vstack((nodes,
                      np.array([[-1 if len(x) == 0 else np.random.choice(x) for x in hops[h].rows[nodes]]
                                for h in hops.keys() if h != -1]),
                      sample_last_hop(hops[-1], nodes)
                      )).T


def to_triplets(sampled_hops, scale_terms):
    """
    Form all valid triplets (pairwise constraints) from a set of sampled nodes in triplets
    Parameters
    ----------
    sampled_hops : array-like, shape [N, K]
       The sampled nodes.
    scale_terms : dict
        The appropriate up-scaling terms to ensure unbiased estimates for each neighbourhood
    Returns
    -------
    triplets : array-like, shape [?, 3]
       The transformed triplets.
    """
    triplets = []
    triplet_scale_terms = []

    for i, j in itertools.combinations(np.arange(1, sampled_hops.shape[1]), 2):
        triplet = sampled_hops[:, [0] + [i, j]]
        triplet = triplet[(triplet[:, 1] != -1) & (triplet[:, 2] != -1)]
        triplet = triplet[(triplet[:, 0] != triplet[:, 1]) & (triplet[:, 0] != triplet[:, 2])]
        triplets.append(triplet)

        triplet_scale_terms.append(scale_terms[i][triplet[:, 1]] * scale_terms[j][triplet[:, 2]])

    return np.row_stack(triplets), np.concatenate(triplet_scale_terms)

class struct_loader_g2g(torch.utils.data.Dataset):
    def __init__(self, G, K):
        print("Preparing to feed edges")
        self.G = G
        self.K = K
        
        train_adj = nx.to_scipy_sparse_matrix(G)
        hops = get_hops_g2g(train_adj, K)

        scale_terms = {h if h != -1 else max(hops.keys()) + 1: hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1 for h in hops}
    
        samples = to_triplets(sample_all_hops(hops), scale_terms)
        
        self.pairs = samples[0]
        self.scales = samples[1]

    def __getitem__(self, index):
        return (self.pairs[index], self.scales[index])

    def __len__(self):
        return self.pairs.shape[0]

def load_dataset(file_name):
    """Load a graph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph


def edge_cover(A):
    """
    Approximately compute minimum edge cover.
    Edge cover of a graph is a set of edges such that every vertex of the graph is incident
    to at least one edge of the set. Minimum edge cover is an  edge cover of minimum size.
    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix
    Returns
    -------
    edges : array-like, shape [?, 2]
        The edges the form the edge cover
    """

    N = A.shape[0]
    d_in = A.sum(0).A1
    d_out = A.sum(1).A1

    # make sure to include singleton nodes (nodes with one incoming or one outgoing edge)
    one_in = np.where((d_in == 1) & (d_out == 0))[0]
    one_out = np.where((d_in == 0) & (d_out == 1))[0]

    edges = []
    edges.append(np.column_stack((A[:, one_in].argmax(0).A1, one_in)))
    edges.append(np.column_stack((one_out, A[one_out].argmax(1).A1)))
    edges = np.row_stack(edges)

    edge_cover_set = set(map(tuple, edges))
    nodes = set(edges.flatten())

    # greedly add other edges such that both end-point are not yet in the edge_cover_set
    cands = np.column_stack(A.nonzero())
    for u, v in cands[d_in[cands[:, 1]].argsort()]:
        if u not in nodes and v not in nodes and u != v:
            edge_cover_set.add((u, v))
            nodes.add(u)
            nodes.add(v)
        if len(nodes) == N:
            break

    # add a single edge for the rest of the nodes not covered so far
    not_covered = np.setdiff1d(np.arange(N), list(nodes))
    edges = [list(edge_cover_set)]
    not_covered_out = not_covered[d_out[not_covered] > 0]

    if len(not_covered_out) > 0:
        edges.append(np.column_stack((not_covered_out, A[not_covered_out].argmax(1).A1)))

    not_covered_in = not_covered[d_out[not_covered] == 0]
    if len(not_covered_in) > 0:
        edges.append(np.column_stack((A[:, not_covered_in].argmax(0).A1, not_covered_in)))

    edges = np.row_stack(edges)

    # make sure that we've indeed computed an edge_cover
    assert A[edges[:, 0], edges[:, 1]].sum() == len(edges)
    assert len(set(map(tuple, edges))) == len(edges)
    assert len(np.unique(edges)) == N

    return edges


def batch_pairs_sample(A, nodes_hide):
    """
    For a given set of nodes return all edges and an equal number of randomly sampled non-edges.
    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix
    Returns
    -------
    pairs : array-like, shape [?, 2]
        The sampled pairs.
    """
    A = A.copy()
    undiricted = (A != A.T).nnz == 0

    if undiricted:
        A = sp.triu(A, 1).tocsr()

    edges = np.column_stack(A.nonzero())
    edges = edges[np.in1d(edges[:, 0], nodes_hide) | np.in1d(edges[:, 1], nodes_hide)]

    # include the missing direction
    if undiricted:
        edges = np.row_stack((edges, np.column_stack((edges[:, 1], edges[:, 0]))))

    # sample the non-edges for each node separately
    arng = np.arange(A.shape[0])
    not_edges = []
    for nh in nodes_hide:
        nn = np.concatenate((A[nh].nonzero()[1], A[:, nh].nonzero()[0]))
        not_nn = np.setdiff1d(arng, nn)

        not_nn = np.random.permutation(not_nn)[:len(nn)]
        not_edges.append(np.column_stack((np.repeat(nh, len(nn)), not_nn)))

    not_edges = np.row_stack(not_edges)

    # include the missing direction
    if undiricted:
        not_edges = np.row_stack((not_edges, np.column_stack((not_edges[:, 1], not_edges[:, 0]))))

    pairs = np.row_stack((edges, not_edges))

    return pairs
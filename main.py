# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 00:46:45 2020

@author: Yongmin Shin
"""
import os.path as osp
import torch
import networkx as nx
import time
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import from_networkx

from utils import preprocess
from utils import get_hops, struct_loader_ver2
from utils import node_split
from utils import get_auc_new, get_f1_new, get_nmi_new
from utils import constkNNGraph_new

from model import GraphSAGE1layer

setting = {}

setting['dataset'] = 'Cora'
setting['PATH_log'] = 'D:/Experiments/Incomplete_coldstart_1.6/log'

setting['node_sampling_ratio'] = 0.9
setting['edge_sampling_ratio'] = 1
setting['feature_sampling_ratio'] = 1

setting['model'] = 'graphsage'
setting['algorithm'] = 'knn'

# Parameter setting
setting['output_dim'] = 64
setting['batch_size'] = 2048

setting['K'] = 2

# Set directory
dataset_name = 'Cora'
assert dataset_name in ['Cora', 'CiteSeer', 'PubMed']

path = osp.join(osp.dirname(osp.realpath('main.py')), '..', 'data', dataset_name)
dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
setting['device'] = str(device)

cos = torch.nn.CosineSimilarity()

def phi(cossim, gamma, b):
    return torch.log(1 + torch.exp(-gamma * cossim + b)) / gamma

def alpha(node0, node1, graph, b):
    batch = torch.cat((node0.view(-1,1), node1.view(-1,1)), axis = 1).cpu().numpy()

    pathlist = torch.Tensor([nx.shortest_path_length(graph, pair[0], pair[1]) for pair in batch])

    return torch.exp(b/pathlist)

setting['is_NS'] = False
setting['negative_ratio'] = 1
setting['lcc'] = True
setting['layer'] = 1
setting['k'] = 3


graph_raw, feature_raw, nodelabel_raw = preprocess(data, lcc = setting['lcc'])
setting['input_dim'] = feature_raw.shape[1]

def main():
    setting['alpha'] = 4
    setting['b'] = 0
    setting['beta'] = 1
    setting['gamma'] = 3

    test_portion = 1 - setting['node_sampling_ratio']
    val_portion = 0.05
    graph_package, total_index, feature, node_label, val_package, test_package = node_split(graph_raw, feature_raw, nodelabel_raw, val_portion, test_portion)
    
    pos_val_edge = val_package[0]
    neg_val_edge = val_package[1]
    pos_test_edge = test_package[0]
    neg_test_edge = test_package[1]
    
    train_graph = graph_package[0]
    val_graph = graph_package[1]
    test_graph = graph_package[2]
    
    knn_graphs, input_features = constkNNGraph_new(feature, len(train_graph), len(val_graph) - len(train_graph), len(test_graph) - len(val_graph), setting['k'])
    
    train_knn, val_knn, test_knn = knn_graphs
    train_feature, val_feature, test_feature = input_features
    
    train_knn_data = from_networkx(train_knn)
    train_knn_data.x = torch.Tensor(train_feature)
    train_knn_data.edge_index, _ = add_self_loops(train_knn_data.edge_index)
    
    val_knn_data = from_networkx(val_knn)
    val_knn_data.x = torch.Tensor(val_feature)
    val_knn_data.edge_index, _ = add_self_loops(val_knn_data.edge_index)
    
    test_knn_data = from_networkx(test_knn)
    test_knn_data.x = torch.Tensor(test_feature)
    test_knn_data.edge_index, _ = add_self_loops(test_knn_data.edge_index)
    
    train_loader = NeighborSampler(train_knn_data.edge_index, node_idx = None,sizes=[-1] * setting['layer'],
                                   batch_size=len(train_knn), shuffle = True)
    val_loader = NeighborSampler(val_knn_data.edge_index, node_idx = None,sizes=[-1] * setting['layer'],
                                   batch_size=len(val_knn), shuffle = True)
    test_loader = NeighborSampler(test_knn_data.edge_index, node_idx = None,sizes=[-1] * setting['layer'],
                                   batch_size=len(test_knn), shuffle = True)
    
    train_knn_data = train_knn_data.to(device)
    val_knn_data = val_knn_data.to(device)
    test_knn_data = test_knn_data.to(device)
    
    # Prep for experiment
    if setting['layer'] == 1:
        model = GraphSAGE1layer(setting['input_dim'], setting['output_dim'], setting['layer'])
    else:
        model = GraphSAGE(setting['input_dim'], setting['output_dim'], setting['layer'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)

    loss_vanilla = []
    

    with torch.no_grad():
        model.eval()
        if setting['layer'] == 1:
            for batch_size, n_id, adjs in test_loader:
                adjs = adjs.to(device)
                totaloutput_test = model(test_knn_data.x, adjs).cpu().numpy()
            for batch_size, n_id, adjs in val_loader:
                adjs = adjs.to(device)
                totaloutput_val = model(val_knn_data.x, adjs).cpu().numpy()
            for batch_size, n_id, adjs in train_loader:
                adjs = adjs.to(device)
                totaloutput_train = model(train_knn_data.x, adjs).cpu().numpy()
        else:
            for batch_size, n_id, adjs in test_loader:
                adjs = [adj.to(device) for adj in adjs]
                totaloutput_test = model(test_knn_data.x, adjs).cpu().numpy()
            for batch_size, n_id, adjs in val_loader:
                adjs = [adj.to(device) for adj in adjs]
                totaloutput_val = model(val_knn_data.x, adjs).cpu().numpy()
            for batch_size, n_id, adjs in train_loader:
                adjs = [adj.to(device) for adj in adjs]
                totaloutput_train = model(train_knn_data.x, adjs).cpu().numpy()

    hops = get_hops(train_graph, setting['K'])
    training_dataset = struct_loader_ver2(train_graph, setting['negative_ratio'], hops, is_NS = setting['is_NS'])

    curr_auc = curr_ap = curr_mic = curr_mac  = curr_km_nmi = curr_sc_nmi = 0
    best_auc = best_ap = best_mic = best_mac  = best_km_nmi = best_sc_nmi = 0

    for epoch in tqdm(range(1, 1 + setting['totepoch'])):
        # Set model to training mode
        model.train()

        start_time = time.time()

        train_gen = torch.utils.data.DataLoader(training_dataset, batch_size = setting['batch_size'], shuffle = True)

        # For each batch
        for batchset in train_gen:
            indexbatch = batchset[0]
            coeff = batchset[1].squeeze().to(device)
            target_batch = indexbatch[:,0].to(device)
            context_batch = indexbatch[:,1].to(device)
            negative_batch = indexbatch[:,2:-1].to(device)
            second_batch = indexbatch[:,-1].to(device)

            batch_size_local = target_batch.size(0)
            negative_depth = 1

            batch_log = []

            optimizer.zero_grad()

            if setting['layer'] == 1:
                for batch_size, n_id, adjs in train_loader:
                    adjs = adjs.to(device)
                    out = model(train_knn_data.x, adjs) # Not elegant but...
            else:
                for batch_size, n_id, adjs in train_loader:
                    adjs = [adj.to(device) for adj in adjs]
                    out = model(train_knn_data.x, adjs)
                    
            # Get embeddings
            target_emb = out[target_batch] 
            context_emb = out[context_batch]
            negative_emb = out[negative_batch]
            second_emb = out[second_batch]

            if setting['negative_ratio'] == 1:
                alpha_weight = alpha(target_batch, negative_batch, train_graph, setting['beta']).to(device)
                enc_neg = alpha_weight * phi(-cos(target_emb, negative_emb.squeeze()), setting['gamma'], setting['b'])
                enc_pos = phi(cos(target_emb, context_emb), setting['gamma'], setting['b']).to(device)
                enc_sec_pos = (coeff * phi(cos(target_emb, second_emb), setting['gamma'], setting['b'])).to(device)
            else:
                alpha_weight = alpha(target_batch.repeat(setting['negative_ratio']), negative_batch, train_graph, setting['beta']).to(device)
                energy_neg = cos(target_emb.repeat(setting['negative_ratio'],1), negative_emb.view(-1,setting['output_dim']))
                enc_neg = alpha_weight * phi(energy_neg, setting['gamma'], setting['b'])
                enc_pos = phi(cos(target_emb, context_emb), setting['gamma'], setting['b']).to(device)
                enc_sec_pos = (coeff * phi(cos(target_emb, second_emb), setting['gamma'], setting['b'])).to(device)
            # Total loss
            loss = (setting['alpha'] * enc_sec_pos.sum() + enc_pos.sum() + enc_neg.sum()) / batch_size_local

            # Calcuate gradient
            loss.backward()
            # Backprop & update pararmeters
            optimizer.step()
            # Log
            batch_log.append(loss.item())
        # Log loss
        loss_val = sum(batch_log)/len(batch_log)

        with torch.no_grad():
            model.eval()
            if setting['layer'] == 1:
                for batch_size, n_id, adjs in test_loader:
                    adjs = adjs.to(device)
                    totaloutput_test = model(test_knn_data.x, adjs).cpu().numpy()
                for batch_size, n_id, adjs in val_loader:
                    adjs = adjs.to(device)
                    totaloutput_val = model(val_knn_data.x, adjs).cpu().numpy()
                for batch_size, n_id, adjs in train_loader:
                    adjs = adjs.to(device)
                    totaloutput_train = model(train_knn_data.x, adjs).cpu().numpy()
            else:
                for batch_size, n_id, adjs in test_loader:
                    adjs = [adj.to(device) for adj in adjs]
                    totaloutput_test = model(test_knn_data.x, adjs).cpu().numpy()
                for batch_size, n_id, adjs in val_loader:
                    adjs = [adj.to(device) for adj in adjs]
                    totaloutput_val = model(val_knn_data.x, adjs).cpu().numpy()
                for batch_size, n_id, adjs in train_loader:
                    adjs = [adj.to(device) for adj in adjs]
                    totaloutput_train = model(train_knn_data.x, adjs).cpu().numpy()

        num_train = len(train_graph)
        num_val = len(val_graph) - len(train_graph)
        val_auc, test_auc, val_ap, test_ap = get_auc_new(totaloutput_val, totaloutput_test, num_train, num_val, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge)
        val_mic, val_mac, test_mic, test_mac = get_f1_new(totaloutput_val, totaloutput_test, node_label, train_graph, val_graph, test_graph)
        val_km_nmi, val_sc_nmi, test_km_nmi, test_sc_nmi = get_nmi_new(totaloutput_val, totaloutput_test, train_graph, val_graph, node_label)

        if val_auc > curr_auc:
            curr_auc = val_auc
            best_auc = test_auc
        if val_ap > curr_ap:
            curr_ap = val_ap
            best_ap = test_ap
        if val_mic > curr_mic:
            curr_mic = val_mic
            best_mic = test_mic
        if val_mac > curr_mac:
            curr_mac = val_mac
            best_mac = test_mac
        if val_km_nmi > curr_km_nmi:
            curr_km_nmi = val_km_nmi
            best_km_nmi = test_km_nmi
        if val_sc_nmi > curr_sc_nmi:
            curr_sc_nmi = val_sc_nmi
            best_sc_nmi = test_sc_nmi
            
        print('[Res]', "AP : %.4f, AUC: %.4f, Macro F1: %.4f, Micro F1: %.4f, NMI: %.4f" % (best_ap, best_auc, best_mac, best_mic, best_km_nmi))

if __name__ == '__main__':
    main()
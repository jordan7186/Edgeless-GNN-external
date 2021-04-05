# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:21:32 2020

@author: CSE-190730
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import SGConv, GCNConv, SAGEConv, GINConv


class mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.encoder(x)
    

class G2G(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(G2G, self).__init__()
        self.Extractor1 = nn.Linear(input_dim, hidden_dim)
        #self.Extractor2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.sig = nn.Linear(hidden_dim, output_dim)
        
        torch.nn.init.xavier_normal_(self.Extractor1.weight)
        torch.nn.init.xavier_normal_(self.mu.weight)
        torch.nn.init.xavier_normal_(self.sig.weight)
        
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
    def forward(self, x):
        x = self.Extractor1(x)
        x = self.relu(x)
        #x = self.Extractor2(x)
        #x = self.relu(x)
        
        mu = self.mu(x)
        sig = self.elu(self.sig(x)) + 1
        
        return mu, sig
    
    def get_emb(self, x):
        x = self.Extractor1(x)
        x = self.relu(x)
        #x = self.Extractor2(x)
        #x = self.relu(x)
        
        mu = self.mu(x)
        
        return mu
    
class DEAL_attr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DEAL_attr, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias = True)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias = True)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))

        return x

class DEAL_struct(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DEAL_struct, self).__init__()
        self.linear = nn.utils.weight_norm(nn.Linear(input_dim, output_dim, bias = False))

    def forward(self, x):
        return self.linear(x)

class SGC(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer):
        super(SGC, self).__init__()
        self.layer = layer
        self.conv = SGConv(input_dim, output_dim, K=self.layer)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv(x, edge_index, edge_weight))
        return x

class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer):
        super(GCN, self).__init__()
        assert layer in [1,2,3]
        self.layer = layer

        self.conv1 = GCNConv(input_dim, output_dim, normalize = True)
        if self.layer in [2, 3]:
            self.conv2 = GCNConv(output_dim, output_dim, normalize = True)
        if self.layer == 3:
            self.conv3 = GCNConv(output_dim, output_dim, normalize = True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        if self.layer in [2, 3]:
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
        if self.layer in [3]:
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index, edge_weight)
        return x

class ResGCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer):
        super(ResGCN, self).__init__()
        assert layer in [1,2,3]
        self.layer = layer
        self.initW = nn.Linear(input_dim, output_dim, bias = None)

        self.conv1 = GCNConv(output_dim, output_dim, normalize = True)
        if self.layer in [2, 3]:
            self.conv2 = GCNConv(output_dim, output_dim, normalize = True)
        if self.layer in [3]:
            self.conv3 = GCNConv(output_dim, output_dim, normalize = True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.initW(x)
        x1 = self.conv1(x, edge_index, edge_weight)
        if self.layer == 1:
            xfinal = x1 + x
        if self.layer == 2:
            x1 = x1 +  x
            x1 = F.relu(x1)
            x1 = F.dropout(x1, training=self.training)
            x2 = self.conv2(x1, edge_index, edge_weight)
            xfinal = x2 + x
        if self.layer == 3:
            x1 = x1 +  x
            x1 = F.relu(x1)
            x1 = F.dropout(x1, training=self.training)
            x2 = self.conv2(x1, edge_index, edge_weight)
            x2 = x2 + x
            x2 = F.relu(x2)
            x2 = F.dropout(x2, training=self.training)
            x3 = self.conv3(x2, edge_index, edge_weight)
            xfinal = x3 + x
        return xfinal

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer):
        super(GraphSAGE, self).__init__()

        self.layer = layer

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, output_dim, normalize = True))
        
        if self.layer > 1:
            for _ in range(self.layer-1):
                self.convs.append(SAGEConv(output_dim, output_dim, normalize = True))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.layer - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class GraphSAGE1layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer, act = 'none'):
        super(GraphSAGE1layer, self).__init__()

        self.layer = layer
        
        if act == 'elu':
            self.act = nn.ELU()
        elif act == 'selu':
            self.act = nn.SELU()
        else:
            self.act = nn.Identity()

        self.convs = SAGEConv(input_dim, output_dim, normalize = False)

    def forward(self, x, adjs):
        edge_index = adjs.edge_index
        size = adjs.size
        x_target = x[:size[1]]  # Target nodes are always placed first.
        x = self.convs((x, x_target), edge_index)
        x = self.act(x)
        x = F.normalize(x, p=2., dim=-1)
        return x
'''
class GIN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer):
        super(GIN, self).__init__()
        assert layer in [1]
        self.layer = layer
        
        nn1 = Sequential(Linear(input_dim, output_dim, bias = False))
        self.conv1 = GINConv(nn1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return x
'''    
class GIN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer):
        super(GIN, self).__init__()
        assert layer in [1,2,3]
        self.layer = layer

        nn1 = Sequential(Linear(input_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv1 = GINConv(nn1)

        if self.layer in [2, 3]:
            self.bn2 = torch.nn.BatchNorm1d(output_dim)
            nn2 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
            self.conv2 = GINConv(nn2)

        if self.layer in [3]:
            self.bn3 = torch.nn.BatchNorm1d(output_dim)
            nn3 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
            self.conv3 = GINConv(nn3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        if self.layer in [2, 3]:
            x = F.relu(x)
            x = self.bn2(x)
            x = self.conv2(x, edge_index)
        if self.layer in [3]:
            x = F.relu(x)
            x = self.bn3(x)
            x = self.conv3(x, edge_index)
        return x

class ResGIN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer):
        super(ResGIN, self).__init__()
        assert layer in [1,2,3]
        self.layer = layer
        self.initW = nn.Linear(input_dim, output_dim, bias = None)

        nn1 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv1 = GINConv(nn1)

        if self.layer in [2, 3]:
            nn2 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
            self.conv2 = GINConv(nn2)

        if self.layer in [3]:
            nn3 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
            self.conv3 = GINConv(nn3)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.initW(x)
        x1 = self.conv1(x, edge_index, edge_weight)
        if self.layer == 1:
            xfinal = x1 + x
        if self.layer == 2:
            x1 = x1 +  x
            x1 = F.relu(x1)
            x1 = F.dropout(x1, training=self.training)
            x2 = self.conv2(x1, edge_index, edge_weight)
            xfinal = x2 + x
        if self.layer == 3:
            x1 = x1 +  x
            x1 = F.relu(x1)
            x1 = F.dropout(x1, training=self.training)
            x2 = self.conv2(x1, edge_index, edge_weight)
            x2 = x2 + x
            x2 = F.relu(x2)
            x2 = F.dropout(x2, training=self.training)
            x3 = self.conv3(x2, edge_index, edge_weight)
            xfinal = x3 + x
        return xfinal

class ResGIN1(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer):
        super(ResGIN1, self).__init__()
        assert layer in [1,2,3]
        self.layer = layer
        self.initW = nn.Linear(input_dim, output_dim, bias = None)

        nn1 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv1 = GINConv(nn1)

        if self.layer in [2, 3]:
            self.bn2 = torch.nn.BatchNorm1d(output_dim)
            nn2 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
            self.conv2 = GINConv(nn2)

        if self.layer in [3]:
            self.bn3 = torch.nn.BatchNorm1d(output_dim)
            nn3 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
            self.conv3 = GINConv(nn3)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.initW(x)
        x1 = self.conv1(x, edge_index, edge_weight)
        if self.layer == 1:
            xfinal = x1 + x
        if self.layer == 2:
            x1 = x1 +  x
            x1 = F.relu(x1)
            self.bn2(x1)
            x2 = self.conv2(x1, edge_index, edge_weight)
            xfinal = x2 + x
        if self.layer == 3:
            x1 = x1 +  x
            x1 = F.relu(x1)
            self.bn2(x1)
            x2 = self.conv2(x1, edge_index, edge_weight)
            x2 = x2 + x
            x2 = F.relu(x2)
            self.bn3(x2)
            x3 = self.conv3(x2, edge_index, edge_weight)
            xfinal = x3 + x
        return xfinal

class SGCEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, ae_type):
        super(SGCEncoder, self).__init__()
        self.ae_type = ae_type
        self.conv1 = SGConv(input_dim, 2 * output_dim)
        if self.ae_type in ['GAE']:
            self.conv2 = SGConv(2 * output_dim, output_dim)
        elif self.ae_type in ['VGAE']:
            self.conv_mu = SGConv(2 * output_dim, output_dim)
            self.conv_logstd = GCNConv(2 * output_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.ae_type in ['GAE']:
            return self.conv2(x, edge_index)
        elif self.ae_type in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, ae_type):
        super(GCNEncoder, self).__init__()
        self.ae_type = ae_type
        self.conv1 = GCNConv(input_dim, 2 * output_dim)
        if self.ae_type in ['GAE']:
            self.conv2 = GCNConv(2 * output_dim, output_dim)
        elif self.ae_type in ['VGAE']:
            self.conv_mu = GCNConv(2 * output_dim, output_dim)
            self.conv_logstd = GCNConv(2 * output_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.ae_type in ['GAE']:
            return self.conv2(x, edge_index)
        elif self.ae_type in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
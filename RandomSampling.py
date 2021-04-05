import random
import networkx as nx
from copy import deepcopy
from networkx import is_connected
from networkx import connected_components as ccs
from random import sample
from numpy.random import choice
from scipy.spatial.distance import cosine
import numpy as np

"""
random_sample_vanilla

Node sampling a graph with pre-determined size in a random fashion.
Starts from full graph(=input graph) and randomly deletes a random node.
For each deletion, the function checks whether
the graph after removal is still fully connected.
The resultant graph is therefore still fully connected.
=======================
input
-----------------------
graph: a networkx graph
sample_portion: relative size (in terms of node count) of the graph after the sampling
=======================
output
-----------------------
randomly sampled graph
"""
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

"""
random_sample_vanilla_edge

Edge sampling a graph with pre-determined size in a random fashion.
Starts from full graph(=input graph) and randomly deletes a random node.
For each deletion, the function checks whether
the graph after removal is still fully connected.
The resultant graph is therefore still fully connected.
=======================
input
-----------------------
graph: a networkx graph
sample_size: relative size (in terms of edge count) of the graph after the sampling
=======================
output
-----------------------
randomly sampled graph
"""
def random_sample_vanilla_edge(graph, sample_portion):
    sample_graph = deepcopy(graph)
    if type(sample_portion) == float:
        sample_edge_num = int(len(sample_graph.edges) * sample_portion)
    elif type(sample_portion) == int:
        sample_edge_num = sample_portion
    else:
        raise ValueError("sample_portion must be either int > 0 or 0 < float <= 1")

    print("Sampling edges... sample {} out of {} edges.".format(sample_edge_num, len(graph.edges)))

    deleted_edges = []

    while len(sample_graph.edges) > sample_edge_num:
        edgelist = list(sample_graph.edges)
        delete_edge = sample(edgelist, 1)[0]

        test_graph = deepcopy(sample_graph)
        test_graph.remove_edge(delete_edge[0], delete_edge[1])

        if is_connected(test_graph) == True:
            sample_graph.remove_edge(delete_edge[0], delete_edge[1])
            deleted_edges.append(tuple(sorted([delete_edge[0], delete_edge[1]])))
        else:
            test_graph = deepcopy(sample_graph)

        del test_graph

    return sample_graph, deleted_edges

"""
random_sample_features

Random feature sampling of the feature matrix.
Does not alter the nodes outside LCC(Largest Connected Component) via mask.
Prevents the case there row without ones are generated.
=======================
input
-----------------------
feature: original feature matrix
mask: row mask for LCC
sample_portion: relative size (in terms of number of one's) after the sampling
=======================
output
-----------------------
sparser feature matrix
"""
def random_sample_features_old(feature, sample_portion, maskindicator):
    # New version: does not care on the number of ones
    feature_matrix = deepcopy(feature)
    row_fulllist = np.array(range(feature.shape[0]))
    col_fullist = np.array(range(feature.shape[1]))

    if type(sample_portion) == float:
        target_count = int(feature_matrix.size * (1 - sample_portion))
    elif type(sample_portion) == int:
        target_count = feature_matrix.size - sample_portion
    else:
        raise ValueError("sample_portion must be either int > 0 or 0 < float <= 1")

    print("Sampling feature matrix... sample {} out of {}.".format(int(target_count), feature_matrix.size))

    count = 0

    while count < target_count:
        row_sample = choice(row_fulllist)
        col_sample = choice(col_fullist)

        while feature_matrix[row_sample, col_sample] == maskindicator:
            row_sample = choice(row_fulllist)
            col_sample = choice(col_fullist)

        # Mask with maskindicator
        feature_matrix[row_sample, col_sample] = maskindicator
        count += 1

    return feature_matrix

def random_sample_features(feature, sample_portion, maskindicator):
    feature_new = deepcopy(feature)
    one_position = np.where(feature_new != 0)
    
    sample_num = int((1-sample_portion) * len(one_position[0]))
    sample_pos = np.random.randint(0,len(one_position[0])-1,sample_num)
    sample_cord = (one_position[0][sample_pos],one_position[1][sample_pos])
    
    feature_new[sample_cord[0], sample_cord[1]] = 0
    
    return feature_new
"""
random_sample_community

Randomly samples a graph with pre-determined size.
Starts from full graph(=input graph) and randomly deletes a
selection of nodes. For each deletion, the function checks
whether the graph after removal is still fully connected.
The resultant graph is therefore still fully connected.

Also, it checks whether the # of communities of the sampled graph
remains the same as the original graph.
=======================
input
-----------------------
graph: a networkx graph
sample_size: size of the final sampled graph
community_num: # of communities in the original graph
node_community_map: a numpy array containing the information of
each nodes' community. node_community_map[i] is the community
number of node i.
=======================
output
-----------------------
randomly sampled graph
"""
def random_sample_community(graph, sample_size, community_num, node_community_map):
    sample_graph = deepcopy(graph)

    while len(sample_graph) > sample_size:
        nodelist = list(sample_graph)
        delete_node = sample(nodelist, 1)[0]

        test_graph = deepcopy(sample_graph)
        test_graph.remove_node(delete_node)

        if is_connected(test_graph) == True:
                community = len(set(node_community_map[test_graph]))
                if community == community_num:
                    sample_graph.remove_node(delete_node)
                else:
                    test_graph = deepcopy(sample_graph)
        else:
            test_graph = deepcopy(sample_graph)

    del test_graph

    return sample_graph

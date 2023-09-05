import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import matplotlib.pyplot as plt
import skfmm
from tqdm import tqdm

from config import (
    BOTTLENECK_CHANNEL, PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z, EDGE_DIST_THRESH, WINDOW_SIZE
)

NUM_FEATURES = int(BOTTLENECK_CHANNEL/8)

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def build_graph(vesselness, feature):
    '''
    Build graph from ground truth vessel probabilities
    Vertex sampling: Input image partitioned into non-overlapping regions (patches),
                     within each region, the pixel with maximum vessel probability is sampled as a vertex
    Edge construction: An undirected edge is assigned between a vertex pair 
                       if their geodesic distance is amller than a threshold d
    Vertex features: feature extracted from the last layer of U-Net before output
    '''
    #find local maxima
    features = np.zeros((0, NUM_FEATURES))
    labels = []
    max_pos = []
    x_quan = range(0, vesselness.shape[0], WINDOW_SIZE)
    y_quan = range(0, vesselness.shape[1], WINDOW_SIZE)
    z_quan = range(0, vesselness.shape[2], WINDOW_SIZE)
    for x_idx in x_quan:
        for y_idx in y_quan:
            for z_idx in z_quan:
                cur_win = vesselness[x_idx:x_idx+WINDOW_SIZE, y_idx:y_idx+WINDOW_SIZE, z_idx:z_idx+WINDOW_SIZE]
                if np.sum(cur_win) == 0:
                    pos = (int(x_idx+WINDOW_SIZE/2), int(y_idx+WINDOW_SIZE/2), int(z_idx+WINDOW_SIZE/2))
                    features = np.append(features, feature[pos].detach().cpu().numpy()[np.newaxis,:], axis=0)
                    labels.append(vesselness[pos])
                    max_pos.append(pos)
                else:
                    #print(np.max(cur_win))
                    temp = np.unravel_index(cur_win.argmax(), cur_win.shape)
                    pos = (int(x_idx+temp[0]), int(y_idx+temp[0]), int(z_idx+temp[0]))
                    features = np.append(features, feature[pos].detach().cpu().numpy()[np.newaxis,:], axis=0)
                    labels.append(vesselness[pos])
                    max_pos.append(pos)
        
    #construct graph
    graph = nx.Graph()
    pos = {}

    #add nodes
    for node_idx, (node_x, node_y, node_z) in enumerate(max_pos):
        pos[node_idx] = [node_x, node_y, node_z]
        graph.add_node(node_idx, x=node_x, y=node_y, z=node_z, label=node_idx)
        #print('node label', node_idx, 'pos', (node_x, node_y, node_z), 'added')

    #add edges 
    node_list = list(graph.nodes)
    edges = np.zeros((0, 2))
    for i, n in enumerate(tqdm(node_list)):
        if vesselness[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] == 0:
            continue
        neighbor = vesselness[max(0, graph.nodes[n]['x']-1):min(PATCH_SIZE_X, graph.nodes[n]['x']+2), \
                              max(0, graph.nodes[n]['y']-1):min(PATCH_SIZE_Y, graph.nodes[n]['y']+2), \
                              max(0, graph.nodes[n]['z']-1):min(PATCH_SIZE_Z, graph.nodes[n]['z']+2)]

        if np.mean(neighbor) < 0.1:
            continue

        phi = np.ones_like(vesselness)
        phi[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] = -1
        #print(np.unique(phi))
        tt = skfmm.travel_time(phi, vesselness, narrow=EDGE_DIST_THRESH)

        for n_comp in node_list[i+1:]:
            geo_dist = tt[graph.nodes[n_comp]['x'], graph.nodes[n_comp]['y'], graph.nodes[n_comp]['z']] #travel time
            if geo_dist < EDGE_DIST_THRESH:
                #print(geo_dist)
                graph.add_edge(n, n_comp, weight=EDGE_DIST_THRESH/(EDGE_DIST_THRESH+geo_dist))
                edges = np.append(edges, np.array([[n, n_comp]]), axis=0)
                #print('An edge between', 'node', n, '&', n_comp, 'is constructed')
    
    labels = np.array(labels)
    labels = encode_onehot(labels)
    labels = torch.LongTensor(np.where(labels)[1])

    features = sp.csr_matrix(features, dtype=np.float32)
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #build symmetric adjacency matrix
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))

    idx_train = np.arange(0, len(max_pos))
    idx_train = torch.LongTensor(idx_train)

    return adj, features, labels, idx_train


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


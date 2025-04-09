import torch
import torch.nn as nn
from collections import deque
from models import GCN, GAT

from typing import Tuple
import yaml

import pandas as pd
import math
from utils.misc import get_degree

def get_partial_deg2_bb(bb, center_idx, max_deg):
    features = torch.zeros((max_deg**2, bb.X.shape[1])).cuda()
    features[:, :] = float('nan')
    former_idxs = torch.zeros(max_deg**2).cuda()
    former_idxs[:] = float('nan')
    
    neighbor_idxs = torch.nonzero(bb.A[center_idx].bool() & (torch.arange(len(bb.A))!=center_idx).cuda())[:, 0]
    features[:max_deg*len(neighbor_idxs):max_deg] = bb.X[neighbor_idxs]
    former_idxs[:max_deg*len(neighbor_idxs):max_deg] = neighbor_idxs
    
    for i, idx in enumerate(neighbor_idxs):
        second_neighbor_idxs = torch.nonzero(bb.A[idx].bool() & ~
            torch.isin(
                torch.arange(len(bb.A)).cuda(),
                torch.tensor([idx, center_idx]).cuda()
            )
        )[:, 0]
        features[max_deg*i+1:max_deg*i+1+len(second_neighbor_idxs)] = bb.X[second_neighbor_idxs]
        former_idxs[max_deg*i+1:max_deg*i+1+len(second_neighbor_idxs)] = second_neighbor_idxs

    return features, torch.cat((torch.tensor([center_idx]).cuda(),former_idxs))

def properly_structure_deg2_grouped(args, deg2s, former_idxs, max_degree, feat_dim):

    centers = deg2s[:, 0, :]
    centers = centers.cuda()
    center_idxs = former_idxs[:, 0]
    next = deg2s[:, 1:, :]
    degrees = get_degree(args, next)
    to_sort = torch.cat((degrees.unsqueeze(2), next), dim=-1)
    list = []
    idx_list = []

    for i in range(deg2s.shape[0]): 
        
        rows_as_lists = to_sort[i].tolist()
        part0 = []
        idxs_list = former_idxs[i, 1:].tolist()
        new_former_idxs = []
        
        for j in range(max_degree):
            
            relevant_rows = rows_as_lists[max_degree*j+1:max_degree*(j+1)]
            relevant_idxs = idxs_list[max_degree*j+1:max_degree*(j+1)]
            
            sorted_indices1 = sorted(range(len(relevant_rows)), key=lambda i: relevant_rows[i], reverse=True)
            relevant_rows = [relevant_rows[k] for k in sorted_indices1]
            relevant_idxs = [relevant_idxs[k] for k in sorted_indices1]

            part0 += [[rows_as_lists[max_degree*j]] + relevant_rows]
            new_former_idxs += [[idxs_list[max_degree*j]] + relevant_idxs] 

        sorted_indices0 = sorted(range(len(part0)), key=lambda i: part0[i], reverse=True)
        
        part0 = [part0[j] for j in sorted_indices0]     
        new_former_idxs = [new_former_idxs[j] for j in sorted_indices0]
        
        sorted_tensor = torch.tensor(part0).reshape(max_degree**2, feat_dim+1)
        sorted_idxs = torch.tensor(new_former_idxs).reshape(max_degree**2)
        list.append(sorted_tensor[:, 1:])
        idx_list.append(sorted_idxs)
        
    all_sorted = torch.stack(list).cuda()
    idx_sorted = torch.stack(idx_list).cuda()
    assert(all_sorted.shape[1]==max_degree**2)
    assert(all_sorted.shape[2]==feat_dim)    
    final = torch.cat((centers.unsqueeze(1), all_sorted), dim = 1)
    final_idxs = torch.cat((center_idxs.unsqueeze(1), idx_sorted), dim=1)

    return final, final_idxs


def tensor2d_to_tuple(tensor: torch.Tensor) -> Tuple:
    return tuple(map(tuple, tensor.tolist()))

def check_for_small_cycle(adj):
    adjacency_matrix = torch.tensor(adj)
    num_nodes = adjacency_matrix.size(dim=0)
    for i in range(num_nodes):
        adjacency_matrix[i][i] = 0
    A2 = torch.matmul(adjacency_matrix, adjacency_matrix)
    A3 = torch.matmul(A2, adjacency_matrix)
    
    # Trace of A^3 (sum of diagonal elements)
    trace_A3 = torch.trace(A3)
    if trace_A3 > 0:
        return True

    # Check for cycle of length 4
    A4 = torch.matmul(A2, A2)
    
    for i in range(num_nodes):
        sum = 0
        for k in range(num_nodes):
            if k!=i:
                sum+=A2[i][k]
            if k==i:
                sum+=A2[i][i]*A2[i][i]
        if A4[i][i]>sum:
            return True
        
    return False    

def bfs_shortest_path_length(adj_matrix, start_node, end_node):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    distance = [float('inf')] * num_nodes
    queue = deque([(start_node, 0)])  # Queue stores tuples of (node, current_distance)
    visited[start_node] = True
    distance[start_node] = 0
    
    while queue:
        current_node, current_distance = queue.popleft()
        
        if current_node == end_node:
            return current_distance
        
        for neighbor, is_connected in enumerate(adj_matrix[current_node]):
            if is_connected and not visited[neighbor]:
                visited[neighbor] = True
                new_distance = current_distance + 1
                distance[neighbor] = new_distance
                queue.append((neighbor, new_distance))
    
    return float('inf')


def is_connected(adjacency_matrix):
    num_nodes = adjacency_matrix.size(0)
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    def dfs(node):
        visited[node] = True
        neighbors = adjacency_matrix[node].nonzero(as_tuple=False).squeeze()
        if neighbors.dim() == 0:
            return
        for neighbor in neighbors:
            if not visited[neighbor]:
                dfs(neighbor)
    dfs(0)
    return visited.all().item()

def get_model(model_args,feat_dim,num_cats):
    if model_args['model'] == 'gcn':
        if model_args['act'] == 'relu':
            activation = nn.ReLU
        elif model_args['act'] == 'sigmoid':
            activation = nn.Sigmoid
        elif model_args['act'] == 'gelu':
            activation = nn.GELU
        else:
            raise NotImplementedError(f'No activation called {model_args["act"]} available')
        model = GCN(
            feat_dim,
            model_args['hidden_size'],
            model_args['node_embedding_dim'],
            model_args['dropout'],
            model_args['readout_hidden_dim'],
            model_args['graph_embedding_dim'],
            num_cats,
            model_args['n_layers_backbone'],
            model_args['n_layers_readout'],
            activation,
            model_args['graph_classification'],
        )
        return model
    elif model_args['model'] == 'gat':
        model = GAT(
            model_args['n_layers_backbone'], 
            [2 for _ in range(model_args['n_layers_backbone'])], 
            [feat_dim]+[model_args['hidden_size'] for _ in range(model_args['n_layers_backbone'])], 
            num_cats, 
            model_args['n_layers_readout'],
            dropout=0.0
        )
        return model
    else:
        raise ValueError(f'Model {model_args["model"]} is not supported, select either "gcn", or "gat".') 
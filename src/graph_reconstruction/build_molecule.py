from collections.abc import Iterable

import torch
import numpy as np
from collections import deque
from utils import NewBuildingBlocks
from utils import normalize_adjacency, possible_feature_values
from utils import get_edges_37, get_degree
from graph_reconstruction.bb_joining import join_deg_2_bb_list_with_cycles_gpu, join_ext_bb_list_with_cycles_gpu

import time

def revert_one_hot_encoding(X):
    lists = sum(possible_feature_values(), [])
    assert(len(lists)==140)
    list_alls = []
    
    for ind in range(len(lists)):
        if X[ind]==1:
            list_alls.append(lists[ind])
        
    return tuple(list_alls)

def check_tensors_within_tolerance(list1, list2, tolerance=1e-4):
    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same length")
    
    for tensor1, tensor2 in zip(list1, list2):

        # Ensure both tensors are of the same shape
        if tensor1.shape != tensor2.shape:
            raise ValueError("All corresponding tensors must have the same shape")
        
        # Compute the absolute difference
        diff = torch.abs(tensor1 - tensor2)
        
        # Check if all differences are less than or equal to the tolerance
        if not torch.all(diff <= tolerance):
            return False
    
    return True


def build_molecule_from_bbs_DFS(args, model_args, scorer, DFS_max_depth, bbs, backup_bbs, deg, model, criterion, gt_gradient, max_degree, feat_dim, bbs_prob=None, backup_bb_probs = None, mol=None, upper_bound_atoms = None, feature_onehot_encoding=None, time_limit_per_branch=900, time_limit_global=1800, grad_limit=500, gt_fms=None, gt_ams=None, gt_ls=None): # Here deg is the degree building blocks from which we'll be recovering the molecule. For now it seems we will be able to recoer deg 1
    
    total_start_time = time.time()
    
    best_recon = None
    best_recon_dist = np.inf
    
    if bbs_prob==None: 
        bbs = NewBuildingBlocks.from_molecule(mol,deg)
    else:
        bbs_sorted = sorted(zip(bbs_prob,bbs), key=lambda x: x[0], reverse=False)
        bbs = [x[1] for x in bbs_sorted]
        
        backup_bbs_sorted = sorted(zip(backup_bb_probs, backup_bbs), key = lambda x: x[0], reverse = False)
        backup_bbs = [x[1] for x in backup_bbs_sorted]

    for use_backups in [False, True]:
        
        if use_backups:
            print('Previous building failed, now using backups')
            
        print('/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/')
        
        start_time = time.time()
        for k, beg in enumerate(bbs[::-1][:3]):
            
            DFS_steps=1
            
            if len(beg.connections) == 0:
                continue
            
            first = beg.connections[0]

            print(f'We start building beginning {k+1} from atom {first}', flush=True)
            
            possible_beginnings_ext = join_deg_2_bb_list_with_cycles_gpu(args, beg, first, backup_bbs if use_backups else bbs, max_degree, feat_dim)
            possible_beginnings_ext.reverse()
            possible_beginnings_ext = possible_beginnings_ext[:max_degree]

            if len(possible_beginnings_ext) == 0:
                continue 

            enum = 0
            max_index_used = 0
            
            list_of_edges_DFS_tree = []
                        
            adding_children = [0]
            
            for i, t in enumerate(possible_beginnings_ext): # We have to iterate over all possibilities for the beginning in order not to miss the corrct solution
                
                enum+=1
                
                all_possible=[(t,2,max_index_used+1, grad_limit)]
                list_of_edges_DFS_tree.append((0,max_index_used+1))
                max_index_used += 1

                while len(all_possible)!=0: # We preform a simple DFS to track all possibilities

                    top = all_possible[0][0]
                    depth = all_possible[0][1]
                    parent_index = all_possible[0][2]
                    grad_limit_iter = all_possible[0][3]
                    all_possible.pop(0)
                    
                    adding_children.append(parent_index)

                    DFS_steps+=1
                    
                    if len(top.connections)==0 or time.time() - start_time > time_limit_per_branch or depth>DFS_max_depth or (upper_bound_atoms is not None and len(top.A)>upper_bound_atoms): # If the current has no dangling nodes, we have reached a possible solution and are adding it to the list of sols 
                        if len(top.connections) == 0:
                            print(f'DFS terminated with graph of size {len(top.A)} at depth {depth}', flush=True)
                        if grad_limit_iter >= 1:
                            l2_diff = scorer.compare_gradients(normalize_adjacency(args, model_args, torch.tensor(top.A)), torch.tensor(top.X))
                            grad_limit_iter -= 1
                            
                            if l2_diff<1e-6:
                                
                                return l2_diff, (top.A, top.X), depth, DFS_steps
                            
                            elif l2_diff < best_recon_dist:
                                
                                best_recon_dist = l2_diff
                                best_recon = (top.A, top.X)

                        continue

                    for i in range(len(top.A)):
                        deg1 = sum(top.A[i])-1
                        if feature_onehot_encoding is False:
                            deg_gt = top.X[i][2]
                        else:
                            deg_gt = get_degree(args, top.X[i])

                        if deg1>deg_gt:
                            continue
                    
                    i = top.connections[0] # Again, doesn't matter which end we start adding to, might as well start with [0]

                    res= join_ext_bb_list_with_cycles_gpu(args, top,i,backup_bbs if use_backups else bbs, max_degree, cycles_time_limit=60)

                    if len(res) == 0:
                        if grad_limit_iter >= 1: 
                            l2_diff = scorer.compare_gradients(normalize_adjacency(args, model_args, torch.tensor(top.A)), torch.tensor(top.X))
                            grad_limit_iter -= 1
                            
                            if l2_diff < best_recon_dist:
                                best_recon_dist = l2_diff
                                best_recon = (top.A, top.X)
                        continue
                    
                    for f in res:
                        if f is not None:
                            all_possible.insert(0,(f,depth+1,max_index_used+1, grad_limit_iter/len(res)))
                            max_index_used+=1
                            list_of_edges_DFS_tree.append((parent_index, max_index_used))
                                    
                    if time.time()>total_start_time + time_limit_global:
                        return best_recon_dist, best_recon, None, None
                    
            
            
    print(f'We (DFS) did not reach the true solution, instead found 1 with distance {best_recon_dist}', flush=True)
    return best_recon_dist, best_recon, None, None

def greedy_building(args, model_args, filtered_bbs, feat_dim):
    max_degree = args.max_degree if not 'zpn' in args.dataset else 6
    reshaped_tensor = filtered_bbs.reshape(filtered_bbs.shape[0]*filtered_bbs.shape[1], feat_dim)
    unique_rows = torch.unique(reshaped_tensor, dim=0)
    unique_rows = unique_rows[~torch.all(unique_rows == 0, dim=1)]
    row_dict = {i: unique_rows[i] for i in range(unique_rows.size(0))}
    reverse_dict = {tuple(row.tolist()): key for key, row in row_dict.items()}

    edges_list_keys = []
    for ind_bb in range(filtered_bbs.shape[0]):
        bb = filtered_bbs[ind_bb]
        for edge in get_edges_37(bb, max_degree):
            i1 = edge[0]
            i2 = edge[1]
            edges_list_keys.append((reverse_dict.get(tuple(bb[i1].tolist()), None), reverse_dict.get(tuple(bb[i2].tolist()), None)))
    nn = unique_rows.shape[0]            

    if nn!=1:
        rows, cols = zip(*edges_list_keys)
        row_indices = torch.tensor(rows)
        col_indices = torch.tensor(cols)
        adj_pred = torch.eye(nn,nn)
        adj_pred[row_indices, col_indices] = 1
    else:
        adj_pred = torch.eye(nn)
    
    if not args.directed:
        adj_pred = torch.maximum(adj_pred, adj_pred.T)
    AX = normalize_adjacency(args, model_args, adj_pred).cuda()@unique_rows
    
    pred = adj_pred, unique_rows
    
    return pred
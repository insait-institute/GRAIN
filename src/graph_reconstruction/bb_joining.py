
from collections.abc import Iterable
from typing import List, Dict
from rdkit import Chem
import torch
import itertools
import copy
import time
from utils import get_degree, check_for_small_cycle, get_partial_deg2_bb, properly_structure_deg2_grouped, get_setting, bfs_shortest_path_length
from utils import ExtendedBuildingBlock

def generate_ordered_combinations(N, k):
    positions_combinations = list(itertools.combinations(range(N), k + 1))
    
    results = []
    for positions in positions_combinations:
        arrangement = [-1] * N
        
        pos_0 = 0
        pos_1 = k + 1
        
        for i in range(N):
            if i in positions:
                arrangement[i] = pos_0
                pos_0 += 1
            else:
                arrangement[i] = pos_1
                pos_1 += 1

        
        results.append(tuple(arrangement))
    
    return torch.tensor(results)
 

def check_equal_neighbours_ext(bb1_ext,idx1,bb2,idx2):
    t1 = [bb1_ext.X[i] + (1,) if (bb1_ext.connections_bool()[i]==1 or sum(bb1_ext.A[i])>=3) else bb1_ext.X[i] + (0,) for i in bb1_ext.ext_neighbors([idx1])]
    t2 = [bb2.X[i] + (1,) if (bb2.connections_bool()[i]==1 or sum(bb2.A[i])>=3) else bb2.X[i] + (0,) for i in bb2.neighbors(idx2)]
    
    t1_list = [list(row) for row in t1]
    t2_list = [list(row) for row in t2]
    tensor1 = torch.tensor(t1_list)
    tensor2 = torch.tensor(t2_list)
    tensor1, _ = torch.sort(tensor1, dim=0)
    tensor2, _ = torch.sort(tensor2, dim=0)

    if tensor1.size(dim=0)!=tensor2.size(dim=0):
        return 0
    abs_difference = torch.abs(tensor1 - tensor2)

    if torch.all(abs_difference<0.01) and tensor1.size()==tensor2.size():
        return 1
    else:
        return 0

def coincide_two_nodes(args, adj, fts, list1, list2):
    assert(len(list1) == len(list2))
    
    A = torch.clone(torch.tensor(adj))
    X = torch.clone(torch.tensor(fts))
    if not args.do_ohe:
        connections = [i for i in range(A.size(dim=0)) if A[i].sum()!=X[i][2]+1]
    else:
        connections = [i for i in range(A.size(dim=0)) if A[i].sum()!=get_degree(args, X[i])+1]
        
    # breakpoint()

    num_nodes = A.size(dim=0)
    num_nodes_new = num_nodes - len([x for x in list2 if x is not None])
    
    for i in range(len(list1)):
        ind1 = list1[i]
        ind2 = list2[i]
        if ind1>ind2:
            list1[i], list2[i] = list2[i], list1[i]
        if torch.any(torch.abs(X[ind1]-X[ind2])>0.01):
            print('HELLO')
            return None
    
    # breakpoint()

    for i in list2:
        if i in connections:
            connections = [t for t in connections if t!=i] # Removing the indices to be deleted from the connections list
    
    # breakpoint()

    mask = torch.tensor([i not in list2 for i in range(num_nodes)])

    new_fts = X[mask]
    
    # breakpoint()

    assert(new_fts.size(dim=0) == num_nodes_new)
    
    new_adj = torch.zeros((num_nodes,num_nodes))
    
    new_adj = A.int()
    
    for i in range(len(list1)):
        x = list1[i]
        corr = list2[i]
        r = torch.bitwise_or(new_adj[x],new_adj[corr])
        rt = r.T
        new_adj[x] = torch.clone(r)
        new_adj[:, x] = torch.clone(rt)
    
    for i in range(len(list2)):
        corr = list2[i]
        zero_row = torch.zeros((num_nodes))
        new_adj[corr] = torch.clone(zero_row)
        new_adj[:,corr] = torch.clone(zero_row.T)
    
    # breakpoint()

    before_del_adj = torch.clone(new_adj)
    
    if not args.directed:
        assert(torch.equal(new_adj, new_adj.T))
    
    indices_to_delete = [p for p in range(num_nodes) if torch.all(new_adj[p]==0)]
    assert(set(indices_to_delete) == set(list2))
    
    # breakpoint()

    connections_new = connections.copy()
    
    for i in range(len(connections)):
        connections_new[i] = connections[i] - len([s for s in indices_to_delete if s<connections[i]])
    
    zero_rows = [x for x in range(new_adj.size(dim=0)) if torch.all(new_adj[x]==0)]
    
    # breakpoint()

    assert(set(zero_rows) == set(indices_to_delete))
    
    assert(torch.equal(new_adj, new_adj.T))
    
    mask = torch.any(new_adj != 0, dim=1)
    new_adj = new_adj[mask]
    mask = torch.any(new_adj != 0, dim=0)
    new_adj = new_adj[:, mask]
    assert(new_adj.size(dim=0) == num_nodes_new and new_adj.size(dim=1) == num_nodes_new)
    row_sums = torch.sum(new_adj, dim=1)
    
    assert(torch.all(row_sums >= 2))
    
    all_res = []
    
    if get_setting(args.dataset) == 'chemical' and check_for_small_cycle(new_adj):
        return None
        
    conns_smart = []
    
    for i in range(num_nodes_new):
        deg1 = new_adj[i].sum()-1
        if not args.do_ohe:
            deg_gt = new_fts[i][2]
        else:
            deg_gt = get_degree(args, new_fts[i])
            
        if deg1<deg_gt:
            conns_smart.append(i)
        if not get_setting(args.dataset) == 'citations' and deg1>deg_gt:
            return None
        
    all_res = ExtendedBuildingBlock(args, new_adj,new_fts,tuple(conns_smart))
        
    return all_res
    
def join_ext_bb_list_with_cycles_gpu(args, bb1, center_idx, bb_list, max_deg, cycles_time_limit):
    feat_dim = len(bb1.X[center_idx])
    bb1.to_tensor()
    for bb in bb_list:
        bb.to_tensor()
        
    list_all = []
    valid_bbs = [bb for bb in bb_list if torch.all( torch.abs(bb1.X[center_idx]-bb.X[bb.middle_idx])<0.01)]
    
    if len(valid_bbs) == 0:
        return []
    
    neighbor_features_1, former_idxs_1 = get_partial_deg2_bb(bb1, center_idx, max_deg) 
    neighbor_features_1 = torch.cat((bb1.X[center_idx].unsqueeze(0), neighbor_features_1)).unsqueeze(0)
    neighbor_features_1, former_idxs_1 = properly_structure_deg2_grouped(args, neighbor_features_1, former_idxs_1.unsqueeze(0), max_deg, feat_dim)
    neighbor_features_1 = torch.nan_to_num(neighbor_features_1.squeeze()[1:].reshape(max_deg,max_deg,-1), nan=-20.) # max_deg, max_deg, FOUT
    
    num_not_nan = max_deg - (neighbor_features_1 == -20.).all(dim=(1,2)).sum().cpu().item()
    combs = generate_ordered_combinations(max_deg, num_not_nan - 1)
    combinations_nf1 = neighbor_features_1[combs].unsqueeze(0) # 1, max_deg!, max_deg, max_deg, FOUT
 
    valid_bbs_features = torch.cat([torch.nan_to_num(bb.ordered_X()[0][1:].reshape(max_deg,max_deg,-1).unsqueeze(0), nan=-20.) for bb in valid_bbs],axis=0).unsqueeze(1) # NBBs, 1, max_deg, max_deg, FOUT
    empty_blocks_combs = ((combinations_nf1 + 20.) < 1e-5).all(dim=(3,4),keepdim=True).expand(valid_bbs_features.shape[0], -1, -1, max_deg, valid_bbs_features.shape[-1])
    
    matches = (combinations_nf1==valid_bbs_features) | empty_blocks_combs  # NBBs, max_deg!, max_deg, max_deg, FOUT (checks if blocks match exactly, or non-matching entries are nans)
    
    can_join = matches.all(dim=(2,3,4)) # NBBS, max_deg!
    join_mask = can_join.any(1)
    valid_bbs = [bb for i, bb in enumerate(valid_bbs) if join_mask[i]]
    can_join = can_join[join_mask].int()
    
    join_idxs = torch.argmax(can_join, 1)
    matching_blocks = torch.nonzero((~(combinations_nf1[0, join_idxs] == -20.).all(dim=(2,3))).int())
    common_neighbors_id = []
    
    for i in range(len(valid_bbs)):
        assert(len(matching_blocks[matching_blocks[:, 0] == i][:, 1])>0)
        common_neighbors_id.append(matching_blocks[matching_blocks[:, 0] == i][:, 1])
        
    common_neighbors = [valid_bbs[i].ordered_X()[1][1:].reshape(max_deg,max_deg)[common_neighbors_id[i],0].int() for i in range(len(valid_bbs))]
    
    bb1.to_tuple()
    for bb in bb_list:
        try:
            bb.to_tuple()
        except:
            pass
    
    for idx, bb2 in enumerate(valid_bbs):
        
        big_adjacency = torch.tensor(bb1.A)
        big_features = torch.tensor(bb1.X)
        big_conns = copy.deepcopy(bb1.connections)
        
        new_indices = []
        num_added = 0
        
        for i in bb2.neighbors(bb2.middle_idx):
            if i in common_neighbors[idx]:
                continue
            curr = big_adjacency.size(dim=0)
            big_adjacency,big_features = add_node(big_adjacency,big_features,torch.tensor(bb2.X[i]),center_idx)
            num_added+=1
            new_indices.append(big_adjacency.size(dim=0)-1)
            if bb2.connections_bool()[i]==1:
                big_conns = big_conns + (curr,)
            for j in bb2.neighbors(i):
                if j==bb2.middle_idx:
                    continue
                big_adjacency,big_features = add_node(big_adjacency,big_features,torch.tensor(bb2.X[j]),curr)
                num_added+=1
                new_indices.append(big_adjacency.size(dim=0)-1)
                if bb2.connections_bool()[j]==1:
                    big_conns = big_conns + (big_adjacency.size(dim=0)-1,)

        for i in big_conns:
            if big_adjacency[i][center_idx]==1:
                    big_conns = tuple(t for t in big_conns if t!=i)
        
        assert(num_added == big_adjacency.size(dim=0) - len(bb1.A))
        
        
        row_sums = torch.sum(big_adjacency, dim=1)
        assert(torch.all(row_sums >= 2))
                
        poss_lists = [[None] for _ in range(num_added)]

        for i in new_indices:
            for j in range(min(new_indices)):
                if torch.all(torch.abs(big_features[i]-big_features[j])<0.01) and bfs_shortest_path_length(big_adjacency, i, j)>=5:
                    poss_lists[i-min(new_indices)].append(j)

        combinations = list(itertools.product(*poss_lists))
                
        start_cycles = time.time()
        

        
        if get_setting(args.dataset) == 'citations':
            
            base = ExtendedBuildingBlock(args, big_adjacency,big_features,big_conns)

            while torch.unique(torch.tensor(base.X), dim=0).shape[0]!=torch.tensor(base.X).shape[0]:
                
                ft = torch.tensor(base.X)
                equal_matrix = (ft[:, None] == ft).all(dim=-1)
                pairs = torch.nonzero(torch.triu(equal_matrix, diagonal=1), as_tuple=False)
                pairs = [(x[0].item(),x[1].item()) for x in pairs]

                selected = []
                used = []
                for pair in pairs:
                    a = pair[0]
                    b = pair[1]
                    if a not in used and b not in used:
                        selected.append((a, b))
                        if a not in used: used.append(a)
                        if b not in used: used.append(b)

                list1 = [x[0] for x in selected]
                list2 = [x[1] for x in selected]
                test = coincide_two_nodes(base.A, base.X, list1, list2)
                if test is None:
                    breakpoint()
                base = test

            list_all.append(base)


        else:

            for combo in combinations:
            
                if time.time() > start_cycles + cycles_time_limit:
                    break
                
                list1 = new_indices
                list2 = combo
                assert(len(list1) == len(list2))
                none_in_list2 = [i for i in range(len(list2)) if list2[i] is None]
                list1 = [list1[i] for i in range(len(list1)) if i not in none_in_list2]
                list2 = [list2[i] for i in range(len(list2)) if list2[i] is not None]

                if len(list1)!=len(set(list1)) or len(list2)!=len(set(list2)):
                    continue
                assert(len(list1) == len(list2))
                if len(list2)!=len(set(list2)):
                    continue
                
                all_res = coincide_two_nodes(args, big_adjacency, big_features, list1, list2)


                if all_res is not None:
                    list_all.append(all_res)

    return list_all
    
def add_node(big_adjacency, big_features, x_to_add, neighbor):
    
    x_to_add = x_to_add.reshape(1,int(x_to_add.size(dim=0)))
    new_big_features = torch.cat((big_features,x_to_add),dim=0)
    curr = big_adjacency.size(dim=0)
    row_to_add = torch.zeros(1,curr)
    row_to_add[0][neighbor]=1
    new_big_adjacency = torch.cat((big_adjacency,row_to_add),dim=0)
    col_to_add = torch.zeros(curr+1,1)
    col_to_add[curr][0] = 1
    col_to_add[neighbor][0]= 1
    new_big_adjacency = torch.cat((new_big_adjacency,col_to_add),dim=1)
    return new_big_adjacency,new_big_features 


def check_deg_2_neighborhood(bb_ext, ind, bb):
    
    if torch.any( torch.abs(torch.tensor(bb.X[bb.middle_idx]) - torch.tensor(bb_ext.X[ind])))>0.01:
        return 0
    
    if check_equal_neighbours_ext(bb_ext,ind,bb,bb.middle_idx)==0:
        return 0

    assert(len(bb_ext.ext_neighbors([ind])) == len(bb.center_neighbors()))

    list_taken = []
        
    for n1 in bb_ext.ext_neighbors([ind]):
        found = False
        for n2 in bb.center_neighbors():
            if torch.any( torch.abs(torch.tensor(bb_ext.X[n1])-torch.tensor(bb.X[n2])))>0.01:
                continue
            if n2 in list_taken:
                continue
            if check_equal_neighbours_ext(bb_ext, n1, bb, n2):
                list_taken.append(n2)
                found = True
                break
        if found==False:
            return 0
    
    return 1

def join_deg_2_bb_list_with_cycles_gpu(args, bb1, center_idx, bb_list, max_deg, feat_dim):
    
    bb1.to_tensor()
    for bb in bb_list:
        bb.to_tensor()
        
    list_all = []
    valid_bbs = [bb for bb in bb_list if torch.all( torch.abs(bb1.X[center_idx]-bb.X[bb.middle_idx])<0.01)]
    
    print(f'Valid bbs: {len(valid_bbs)}')

    if len(valid_bbs)==0:
        return []
    
    neighbor_features_1, former_idxs_1 = get_partial_deg2_bb(bb1, center_idx, max_deg) 
    neighbor_features_1 = torch.cat((bb1.X[center_idx].unsqueeze(0), neighbor_features_1)).unsqueeze(0)
    neighbor_features_1, former_idxs_1 = properly_structure_deg2_grouped(args, neighbor_features_1, former_idxs_1.unsqueeze(0), max_deg, feat_dim)
    neighbor_features_1 = torch.nan_to_num(neighbor_features_1.squeeze()[1:].reshape(max_deg,max_deg,-1), nan=-20.) # max_deg, max_deg, FOUT

    num_not_nan = max_deg - (neighbor_features_1 == -20.).all(dim=(1,2)).sum().cpu().item()
    combs = generate_ordered_combinations(max_deg, num_not_nan - 1)
    combinations_nf1 = neighbor_features_1[combs].unsqueeze(0) # 1, max_deg!, max_deg, max_deg, FOUT
                    
    valid_bbs_features = torch.cat([torch.nan_to_num(bb.ordered_X()[0][1:].reshape(max_deg,max_deg,-1).unsqueeze(0), nan=-20.) for bb in valid_bbs],axis=0).unsqueeze(1) # NBBs, 1, max_deg, max_deg, FOUT
    empty_blocks_combs = ((combinations_nf1 + 20.) < 1e-5).all(dim=(3,4),keepdim=True).expand(valid_bbs_features.shape[0], -1, -1, max_deg, valid_bbs_features.shape[-1])
    
    matches = (combinations_nf1==valid_bbs_features) | empty_blocks_combs  # NBBs, max_deg!, max_deg, max_deg, FOUT (checks if blocks match exactly, or non-matching entries are nans)
    
    can_join = matches.all(dim=(2,3,4)) # NBBS, max_deg!
    join_mask = can_join.any(1)
    valid_bbs = [bb for i, bb in enumerate(valid_bbs) if join_mask[i]]
    can_join = can_join[join_mask].int()
    
    join_idxs = torch.argmax(can_join, 1)
    matching_blocks = (~(combinations_nf1[0, join_idxs] == -20.).all(dim=(2,3))).int().argmax(1)
    common_neighbors = [valid_bbs[i].ordered_X()[1][1:].reshape(max_deg,max_deg)[matching_blocks[i],0].int().item() for i in range(len(valid_bbs))]


    bb1.to_tuple()
    for bb in bb_list:
        try:
            bb.to_tuple()
        except:
            pass
    

    for idx in range(len(valid_bbs)):
        
        list_all_new = []
        
        bb2 = valid_bbs[idx]
        
        
        big_adjacency = torch.tensor(bb1.A).clone()
        big_features = torch.tensor(bb1.X).clone()
        big_conns = copy.deepcopy(bb1.connections)
        
        for i in bb2.neighbors(bb2.middle_idx):
            if i==common_neighbors[idx]:
                continue
            curr = big_adjacency.size(dim=0)
            big_adjacency,big_features = add_node(big_adjacency,big_features,torch.tensor(bb2.X[i]),center_idx)
            if bb2.connections_bool()[i]==1:
                big_conns = big_conns + (curr,)
            for j in bb2.neighbors(i):
                if j==bb2.middle_idx:
                    continue
                big_adjacency,big_features = add_node(big_adjacency,big_features,torch.tensor(bb2.X[j]),curr)
                if bb2.connections_bool()[j]==1:
                    big_conns = big_conns + (big_adjacency.size(dim=0)-1,)

        for i in big_conns:
            if sum(big_adjacency[i])>=3:
                big_conns = tuple(t for t in big_conns if t!=i)

        if get_setting(args.dataset) == 'citations':

            base = ExtendedBuildingBlock(args, big_adjacency,big_features,big_conns)

            while torch.unique(torch.tensor(base.X), dim=0).shape[0]!=torch.tensor(base.X).shape[0]:
                
                ft = torch.tensor(base.X)
                equal_matrix = (ft[:, None] == ft).all(dim=-1)
                pairs = torch.nonzero(torch.triu(equal_matrix, diagonal=1), as_tuple=False)
                pairs = [(x[0].item(),x[1].item()) for x in pairs]

                selected = []
                used = []
                for pair in pairs:
                    a = pair[0]
                    b = pair[1]
                    if a not in used and b not in used:
                        selected.append((a, b))
                        if a not in used: used.append(a)
                        if b not in used: used.append(b)

                list1 = [x[0] for x in selected]
                list2 = [x[1] for x in selected]
                test = coincide_two_nodes(args, base.A, base.X, list1, list2)
                if test is None:
                    breakpoint()
                base = test

            list_all_new.append(base)

        else:
            list_all_new.append(ExtendedBuildingBlock(args, big_adjacency,big_features,big_conns))

            num_nodes = big_adjacency.size(dim=0)
            poss = [] # poss will be a list of pairs of nodes that might potentially coincide
            A = big_adjacency.clone()
            X = big_features.clone()
            for i in range(num_nodes):
                for j in range(i):
                    if torch.all(torch.abs(X[i]-X[j])<0.01) and bfs_shortest_path_length(A,i,j)>=5:
                        # The bfs_shortest path restriction comes from the fact we cannot have a cycle of length 3 or 4
                        poss.append((i,j))  


            # When we have just glues the first two bbs we accept that any two nodes could coincide, so we do 2^{number of pairs} checks
            for binary_list in list(itertools.product([0, 1], repeat=len(poss))):
                if all(bit==0 for bit in binary_list):
                    continue
                poss2 = [poss[x] for x in range(len(poss)) if binary_list[x]==1]
                list1 = [pair[0] for pair in poss2]
                list2 = [pair[1] for pair in poss2]

                # list1[i] will coincide with list2[i]
                if len(list1)!=len(set(list1)) or len(list2)!=len(set(list2)): # All entries in both lists should be different
                    continue
                all_res = coincide_two_nodes(args, A, X, list1, list2)
                if all_res is not None:
                    list_all_new.append(all_res)


            for ind, pb in enumerate(list_all_new):
                for i in range(len(pb.A)):
                    if not args.do_ohe:
                        if i not in pb.connections and sum(pb.A[i])!=pb.X[i][2]+1:
                            list_all_new.pop(ind)
                            break
                    else:
                        if i not in pb.connections and sum(pb.A[i])!=get_degree(args, pb.X[i])+1:
                            print(f'Degrees from X: {[get_degree(args, pb.X[i]) for i in range(len(pb.A))]}')
                            print(f'Degrees from A: {[sum(pb.A[i])-1 for i in range(len(pb.A))]}')

                            print('-----------------------------------------------------------------')
                            list_all_new.pop(ind)
                            break
                        
            for ind,candidate in enumerate(list_all_new):
            
                found_bb1 = False
                found_bb2 = False

                for pos in range(len(candidate.A)):
                    if check_deg_2_neighborhood(candidate, pos, bb1):
                        found_bb1 = True
                    if check_deg_2_neighborhood(candidate, pos, bb2):
                        found_bb2 = True
                if found_bb1 is False or found_bb2 is False:
                    list_all_new.pop(ind)
                        
        list_all += list_all_new

    
    return list_all
                    
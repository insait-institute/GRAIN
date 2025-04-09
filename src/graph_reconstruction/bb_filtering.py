import torch
import torch
import time
import numpy as np
from utils import get_relevant_gradients, get_layer_decomp, check_if_in_span, get_degree, are_compatible_pos, features_combined_ultimate, \
    properly_structure_deg1, get_l2_embeddings, get_l3_embeddings
from utils import normalize_adjacency, normalize_features, get_setting
from graph_reconstruction.node_filtering import filter_nodes
from models import GCN, GAT


def filter_bbs(args, model_args, run, model, scorer, criterion, gradients, tol1, tol2, tol3, batch = None, is_gc=True):
   
    gradient_l1, gradient_l2, gradient_l3 = get_relevant_gradients(gradients, model)
    feat_dim = gradient_l1.shape[1]

    if run:
        run["use degree"] = args.use_degree
        run["ranks/rank l1"].append(torch.linalg.matrix_rank(gradient_l1, tol=tol1))
        run["ranks/rank l2"].append(torch.linalg.matrix_rank(gradient_l2, tol=tol2))
        run["ranks/rank l3"].append(torch.linalg.matrix_rank(gradient_l3, tol=tol3))
        
    if args.debug: 
        gt_fms, _, _, _ = batch
        gt_fms_copy = torch.clone(gt_fms)
        _, R_K_l1 = get_layer_decomp(gradient_l1, tol=tol1)
        R_K_l1 = R_K_l1.cuda()
        diffs = check_if_in_span(R_K_l1, gt_fms_copy)
        diffs = torch.log10(diffs)
        diffs = torch.where(diffs == -float('inf'), torch.tensor(-7.0), diffs)
            
        if torch.max(diffs)>-2:
            print('We are going to miss something')
    elif batch is not None:
        raise ValueError('')
    with torch.no_grad():
        global survived
        survived = True

        passed_nodes = filter_nodes(args, gradient_l1, tol1)

        if passed_nodes is not None:
            assert(passed_nodes.shape[1] == feat_dim), "The filtered nodes are of incorrect shape"
            print(f'Filtered {passed_nodes.shape[0]} nodes')

        if passed_nodes is None:
            print('No nodes passed')
            marker, filtered_bbs = None, None
        else:
            marker, filtered_bbs = generate_deg1s(args, model_args, gradients, model, scorer, run, passed_nodes, tol2=tol2, batch=batch if args.debug else None)
        
            if filtered_bbs is not None:
                print(f'Filtered {filtered_bbs.shape[0] if len(filtered_bbs.shape) > 2 else 1} 1-hop building blocks')

        if not marker:
            return False, filtered_bbs, None, None, None, filtered_bbs
            

        return generate_deg2(args, model_args, run, model, scorer, criterion, gradients, filtered_bbs, tol2=tol2, tol3=tol3, batch=batch if args.debug else None, is_gc=is_gc) 

def generate_deg2_candidates_directed(args, filtered_bbs, feat_dim, max_degree=6):
    '''
    Generates candidate degree 2 bbs in the directed setting
    '''
    centers = filtered_bbs[:,0,:]
    A_expanded = centers.view(centers.shape[0], 1, 1, centers.shape[1])    # Shape (2, 1, 1, 3803)
    B_expanded = filtered_bbs.view(1, filtered_bbs.shape[0], filtered_bbs.shape[1], filtered_bbs.shape[2])  # Shape (1, 2, 101, 3803)
    matches = torch.all(A_expanded == B_expanded, dim=3)  # Shape (2, 2, 101)
    # Use torch.nonzero to get indices where matches occur
    match_indices = torch.nonzero(matches, as_tuple=False)  # Shape (num_matches, 3)
    match_indices = match_indices[match_indices[:,2]!=0] # Removing the cases where something equals a center
    is_db = get_degree(args, filtered_bbs)>=1
    is_db[:,0] = False # Centers cannot be dangling bits
    bbs_with_dbs = torch.any(is_db, dim=1)
    bbs_with_dbs = torch.nonzero(bbs_with_dbs, as_tuple=False).squeeze()
    
    if bbs_with_dbs.ndimension()==0:
        bbs_with_dbs = bbs_with_dbs.unsqueeze(0)
    
    list_deg2_built = []
    
    for bb_idx in bbs_with_dbs:
        db_indices = torch.where(is_db[bb_idx])[0]
        current_tensor = filtered_bbs[bb_idx].unsqueeze(0)
        for db in db_indices:
            mask = (match_indices[:, 1] == bb_idx) & (match_indices[:, 2] == db)
            possible_gluings = match_indices[mask, 0]
            to_add = filtered_bbs[possible_gluings][:,1:max_degree,:]
            repeated_current = current_tensor.repeat(possible_gluings.shape[0], 1, 1)
            repeated_add = to_add.unsqueeze(1).repeat(1, current_tensor.shape[0], 1, 1).view(-1, max_degree-1, feat_dim)
            repeated_current[:, max_degree+(db-1)*(max_degree-1)+1:max_degree + (db-1)*(max_degree-1)+1+max_degree-1,:] = repeated_add
            current_tensor = repeated_current
        list_deg2_built.append(current_tensor)
    
    return torch.cat(list_deg2_built, dim=0)

def generate_deg2_candidates_undirected(args, filtered_bbs, X_dangling_bits_filtered, num_nonzero_rows, max_degree, feat_dim):
    '''
    Generates candidate degree 2 bbs in the undirected setting
    '''
    mini_B = 100
    n_bbs = filtered_bbs.shape[0]
    global list_tensors_pos
    list_tensors_pos = [torch.zeros((n_bbs, n_bbs), dtype=torch.int64).cuda() for _ in range(max_degree+1)]

    for pos in range(1,max_degree+1):
    
        sum1=0

        for k in range( (n_bbs + mini_B - 1) // mini_B ):
            st = k * mini_B
            en = (k + 1) * mini_B
            en = min( en, n_bbs )
            i = torch.arange( st, en )
            j = torch.arange( n_bbs )
            i_idx, j_idx = torch.meshgrid(i,j)
            i_idx = i_idx.reshape(-1)
            j_idx = j_idx.reshape(-1)
            try:
                val_pos_1 = are_compatible_pos(args, filtered_bbs[i_idx],num_nonzero_rows[i_idx],filtered_bbs[j_idx], num_nonzero_rows[j_idx], pos)
            except:
                continue

            list_tensors_pos[pos][i_idx,j_idx] = val_pos_1
            sum1+=torch.sum(val_pos_1>0)

    list_tensors_pos = torch.stack([list_tensors_pos[i] for i in range(0,max_degree+1)])
    
    total_deg2 = 0

    xs_db_list = [ [] ]
    yes_stacked_db_list = [ [] ]

    candidate_deg2s = []

    if args.use_degree:
        all0s = torch.where(X_dangling_bits_filtered==0)
        candidate_deg2s.append(filtered_bbs[all0s])

    else: X_dangling_bits_filtered = num_nonzero_rows - 1

    for num_dangling in range(1,max_degree+1):

        all = torch.where(X_dangling_bits_filtered==num_dangling)

        if all[0].shape[0]>=1:
            start = all[0][0]
            end = all[0][-1]
        else:
            xs_db_list.append(torch.empty(0))
            yes_stacked_db_list.append(torch.empty(0))
            continue
        
        xs_db_tensor = torch.full((1,1),-1).cuda()
        ys_db_stacked = torch.full((1,num_dangling), -1).cuda()

        for i in range(start, end+1):

            list_wheres = []
            all_possible = True
            for d in range(1, num_dangling+1):
                list_wheres.append(torch.where(list_tensors_pos[d][i])[0])
                if torch.where(list_tensors_pos[d][i])[0].shape[0]==0:
                    all_possible = False
                    break
                assert(torch.where(list_tensors_pos[d][i])[0].shape[0]>0)

            if all_possible is False:
                continue
            
            tuple_i_hope = torch.meshgrid(*list_wheres)
            stacked_tuple = torch.stack(tuple_i_hope, dim=0)
            tensors_meshgrided = stacked_tuple.T
            tensors_meshgrided = tensors_meshgrided.reshape(-1,num_dangling)

            sorted_rows, _ = torch.sort(tensors_meshgrided, dim=1)

            _, indices = torch.unique(sorted_rows, return_inverse=True, dim=0)
            list_indices = list(indices)
            list_indices = [x.item() for x in list_indices]

            unique_indices = list(set([list_indices.index(x) for x in set(list_indices)]))

            reduced_tensor_meshgrided = tensors_meshgrided[unique_indices]
            num_comb = reduced_tensor_meshgrided.shape[0]

            xs_db_tensor = torch.cat((xs_db_tensor, torch.full((num_comb,1), i).cuda()), dim=0)
            ys_db_stacked = torch.cat((ys_db_stacked, reduced_tensor_meshgrided), dim=0)

            assert(ys_db_stacked.shape[1]==num_dangling)

            total_deg2 += reduced_tensor_meshgrided.shape[0]

        assert(ys_db_stacked.shape[1]==num_dangling)
        xs_db_list.append(xs_db_tensor[1:])
        yes_stacked_db_list.append(ys_db_stacked[1:])

    if total_deg2 == 0:
        return None

    mini_B1 = 1000

    for num_dangling in range(1,max_degree+1):

        xs_idxs = xs_db_list[num_dangling]
        ys_idxs = yes_stacked_db_list[num_dangling]
        total = xs_idxs.shape[0]
        assert(xs_idxs.shape[0] == ys_idxs.shape[0])

        for k in range( (total + mini_B1 - 1) // mini_B1 ):
            st = k * mini_B1
            en = (k + 1) * mini_B1
            en = min( en, total )
            try: candidate_deg2s.append(features_combined_ultimate(xs_idxs[st:en], ys_idxs[st:en], filtered_bbs, list_tensors_pos, max_degree, feat_dim))
            except: continue

    candidate_deg2s = torch.cat(candidate_deg2s, dim=0)
    try:
        candidate_deg2s = torch.nan_to_num(candidate_deg2s, nan=0.0)
    except:
        candidate_deg2s = candidate_deg2s[:500,:,:]
        candidate_deg2s = torch.nan_to_num(candidate_deg2s, nan=0.0)
    
    return candidate_deg2s

def compute_deg2_probabilities_naive(args, filtered_bbs, list_deg2s_passed, diffs_passed, gradient_l2, tol2, is_chemical):
    '''
    Computes the probability score for the general setting (in a simplified manner)
    '''
    final_deg2s = list_deg2s_passed
    final_probs = 1 - diffs_passed

    if (args.naive_build or is_chemical) or final_deg2s.shape[0]>=torch.linalg.matrix_rank(gradient_l2, tol=tol2)/2:
        return True, final_deg2s, final_probs, final_deg2s, final_probs, filtered_bbs
    else:
        return True, filtered_bbs, [1 for _ in range(filtered_bbs.shape[0])], filtered_bbs, [1 for _ in range(filtered_bbs.shape[0])], filtered_bbs

def compute_deg2_probabilities_chemical(args, deg2s_passed, diffs_passed, feat_dim, filtered_bbs):
    '''
    Computes the probability score for the chemical setting
    '''
    final_deg2s = []
    final_probs = []
    filtered_deg2_bbs = []
    all_deg2s = []
    all_probs = []

    for i, bb in enumerate(deg2s_passed):
        bb_died = False

        bb_prob = 0.
        num_average = 1

        for nb in range(7,37):

            if torch.sum(bb[nb])==0:
                break
            
            cb = (nb-2) // 5

            target_central = torch.cat((bb[nb].unsqueeze(0), bb[cb].unsqueeze(0), bb[0].unsqueeze(0), bb[5*cb+2:nb], bb[nb+1:5*cb+7]), dim=0)

            to_sort = torch.cat(( get_degree(args, target_central[2:].unsqueeze(0)).T, target_central[2:]), dim=1)
            rows_as_lists = to_sort.tolist()
            sorted_indices = sorted(range(len(rows_as_lists)), key=lambda i: rows_as_lists[i], reverse=True)
            sorted_target = to_sort[sorted_indices][:,1:]
            total = torch.cat((bb[nb].unsqueeze(0),torch.zeros((36,feat_dim), dtype = torch.float32).cuda()))

            found = False

            max_prob = -np.inf

            for j in range(1,7):

                total_new = torch.clone(total)
                total_new[j] = bb[cb]
                total_new[5*j+2:5*j+7] = sorted_target

                mask_list_deg2s_passed = (deg2s_passed > 0).int()
                mask_total_new = (total_new > 0).int()

                subtracted = mask_list_deg2s_passed - mask_total_new
                non_negative_slices = (subtracted >= 0).all(dim=(1, 2))

                if  torch.sum(non_negative_slices==True)!=0:
                    found = True
                    max_prob = max(max_prob, torch.max(-diffs_passed[non_negative_slices.cpu()]))

            bb_prob+=max_prob

            if not found:
                bb_died = True
            else:
                num_average+=1

        for nb in range(1,7):
            if torch.sum(bb[nb])==0:
                break
            target_central = torch.cat((bb[nb].unsqueeze(0), bb[0].unsqueeze(0), bb[5*nb+2:5*nb+7]), dim=0)
            to_sort = torch.cat(( get_degree(args, target_central[1:].unsqueeze(0)).T, target_central[1:]), dim=1)
            rows_as_lists = to_sort.tolist()
            sorted_indices = sorted(range(len(rows_as_lists)), key=lambda i: rows_as_lists[i], reverse=True)
            sorted_target = torch.cat((bb[nb].unsqueeze(0), to_sort[sorted_indices][:,1:]), dim=0)
            center_index = sorted_indices.index(0)+1
            assert(torch.all(bb[0] == sorted_target[center_index]))
            assert(sorted_target.shape[0]==7)
            extra = torch.zeros((30,feat_dim), dtype = torch.float32).cuda()
            total = torch.cat((sorted_target, extra), dim=0)

            mask_center_nb = torch.ones(6, dtype = torch.bool)
            mask_center_nb[nb-1] = False
            to_add = bb[1:7][mask_center_nb]

            comparison = total[:7] == total[center_index]
            rows_equal_to_specific = comparison.all(dim=1) 
            equal_row_indices = torch.nonzero(rows_equal_to_specific, as_tuple=False).squeeze()
            if equal_row_indices.dim() == 0:
                possible_center_indices = [equal_row_indices.item()]
            else:
                possible_center_indices = equal_row_indices.tolist()


            found = False

            max_prob = -np.inf

            for center_index in possible_center_indices:

                total_copy = torch.clone(total)

                total_copy[5*center_index+2:5*center_index+7] = to_add

                mask_list_deg2s_passed = (deg2s_passed > 0).int()
                mask_total = (total > 0).int()
                subtracted = mask_list_deg2s_passed - mask_total

                non_negative_slices = (subtracted >= 0).all(dim=(1, 2))
                if  torch.sum(non_negative_slices==True)!=0:
                    found = True
                    max_prob = max(max_prob, torch.max( - diffs_passed[non_negative_slices.cpu()]))

            bb_prob+=max_prob

            if not found:
                bb_died = True
            else:
                num_average+=1

        if not bb_died:
            filtered_deg2_bbs.append(bb.unsqueeze(0))      

        if not bb_died:
            final_deg2s.append(bb)
            final_probs.append(bb_prob)

        all_deg2s.append(bb)
        all_probs.append(bb_prob)

    if len(final_deg2s)==0:
        return None, None, None, None, None, None

    final_deg2s = torch.stack(final_deg2s)
    all_deg2s = torch.stack(all_deg2s)

    return True, final_deg2s, final_probs, all_deg2s, all_probs, filtered_bbs    

def test_candidate_deg2_complete(args, candidate_deg2s, scorer, max_degree):
    '''
    Checks whether the generated candidates can produce the same gradients as the client
    '''
    for ind in range(candidate_deg2s.shape[0]):
        bb = candidate_deg2s[ind]
        degs = get_degree(args, bb[max_degree+1:max_degree*max_degree+1])
        dangling_bits = torch.sum(degs>1)
        if args.debug:
            print(f'{dangling_bits} dangling bits')
        if dangling_bits!=0:
            continue
        degrees = get_degree(args, bb.unsqueeze(0)) + 1
        degrees = torch.where(degrees==1, torch.tensor(0.0), degrees)
        result = torch.bmm(degrees.unsqueeze(2), degrees.unsqueeze(1))
        result = torch.pow(result, -0.5)
        result = torch.where(torch.isinf(result), torch.tensor(0.0), result)
        bb_size = max_degree * max_degree + 1
        bb_adj = torch.eye(bb_size, device='cuda')  # Initialize with identity matrix

        # Connect first row and column to node 0
        bb_adj[0, :max_degree + 1] = 1
        bb_adj[:max_degree + 1, 0] = 1

        # Connect internal nodes
        for i in range(1, max_degree + 1):
            start, end = (max_degree - 1) * i + 2, (max_degree - 1) * i + max_degree + 1
            bb_adj[i, start:end] = 1
            bb_adj[start:end, i] = 1

        all_adj = result * bb_adj

        nonzero_rows = torch.nonzero( torch.any(all_adj.squeeze(0) > 0, dim=1) ).squeeze()
        all_adj = all_adj.squeeze(0)[nonzero_rows][:, nonzero_rows].T

        bb_reduced = bb[nonzero_rows]

        if bb_reduced.dim()==1:
            bb_reduced = bb_reduced.unsqueeze(0)
            all_adj = all_adj.unsqueeze(0).unsqueeze(0)

        if scorer.compare_gradients(all_adj, bb_reduced) < 1e-6:
            print('Found building block with ')
            return bb

def generate_deg2(args, model_args, run, model, scorer, criterion, gradients, filtered_bbs, tol2, tol3, batch = None, is_gc=True):
    gradient_l1, gradient_l2, gradient_l3 = get_relevant_gradients(gradients, model)
    feat_dim = gradient_l1.shape[1]
    is_chemical = get_setting(args.dataset) == 'chemical'
    max_degree = args.max_degree if not is_chemical else 6


    sliced_tensor = filtered_bbs[:, 1:, :] # Removed the center from all the deg1 bbs
    if is_chemical:
        relevant_columns = sliced_tensor[:, :, 60:65] # We only take 60:65 because 58 and 59 correspond to nodes of degree 0/1 which cannot be dangling
    if args.dataset == 'citeseer':
        if not args.directed:
            relevant_columns = sliced_tensor[:, :, 3705:] 
        else:
            relevant_columns = sliced_tensor[:, :, 3704:] # In the directed case a dangling bit is anything with positive in-degree

    if args.dataset == 'pokec':
        relevant_columns = sliced_tensor[:, :, -100:] 
    has_one = relevant_columns > 0
    has_one_in_any_column = has_one.any(dim=2)
    X_dangling_bits_filtered = has_one_in_any_column.sum(dim=1) # Extracts the number of dangling bits for every deg 1 bb
    
    row_sums = filtered_bbs.sum(dim=2)
    non_zero_rows = row_sums!=0
    num_nonzero_rows = non_zero_rows.sum(dim=1) # The sizes of all the filtered deg 1 bbs
            

    if not args.use_degree: X_dangling_bits_filtered = num_nonzero_rows - 1

    X_dangling_bits_filtered, sort_indices = X_dangling_bits_filtered.sort(dim=0) 
    filtered_bbs = filtered_bbs[sort_indices, :, :] 
    num_nonzero_rows = num_nonzero_rows[sort_indices] 


    if filtered_bbs.dim()==2:
        filtered_bbs = filtered_bbs.unsqueeze(0)

    if filtered_bbs.dim()==2:
        filtered_bbs = filtered_bbs.unsqueeze(0)

    if args.debug:
        print('Generated and sorted deg 1s')
    
    if args.directed:
        candidate_deg2s = generate_deg2_candidates_directed(args, filtered_bbs, max_degree, feat_dim)

    if not args.directed:
        candidate_deg2s = generate_deg2_candidates_undirected(args, filtered_bbs, X_dangling_bits_filtered, num_nonzero_rows, max_degree, feat_dim)
        if candidate_deg2s is None:
            return None, None, None, None, None, None

    candidate_embeddings = get_l3_embeddings(args, model, candidate_deg2s)
    _, R_K_norm = get_layer_decomp(gradient_l3.double().cpu(), tol=tol3)

    diffs = check_if_in_span(R_K_norm.double().cpu(), candidate_embeddings.double().cpu()).double()
        
    mask_passed = torch.zeros(candidate_deg2s.shape[0], dtype=torch.bool)
    mask_passed[diffs<1e-2] = True
    
    # directly check whether the deg2s with 0 dangling bits are the correct graph
    # and discard them otherwise
    if is_chemical:
        sol_candidate = test_candidate_deg2_complete(args, candidate_deg2s, scorer, max_degree)
        if sol_candidate is not None:
            return False, sol_candidate, None, None, None, filtered_bbs

    diffs_passed = diffs[mask_passed]
    deg2s_passed = candidate_deg2s[mask_passed]


    if not is_chemical:
        return compute_deg2_probabilities_naive(args, filtered_bbs, deg2s_passed, diffs_passed, gradient_l2, tol2, is_chemical)
    else:
        return compute_deg2_probabilities_chemical(args, deg2s_passed, diffs_passed, feat_dim, filtered_bbs)
        

def verify_gt_filter(args, gt_fms, passed_nodes):
    list_correct_idxs = []
    is_chemical = get_setting(args.dataset) == 'chemical'

    if is_chemical:
        ft_norm = torch.clone(gt_fms)
    else:
        ft_norm = normalize_features(args, gt_fms)


    for i, gt_row in enumerate(ft_norm):
        if not torch.any(torch.all(passed_nodes == gt_row, dim=1)):
            print('GT deg 0 not passed')
        else:
            print(f'to {i} corresponds {torch.where(torch.all(passed_nodes == gt_row, dim=1))}')
            list_correct_idxs.append(torch.where(torch.all(passed_nodes == gt_row, dim=1))[0].item())

def generate_deg1s(args, model_args, gradients, model, scorer, run, passed_nodes, tol2=1e-7, batch=None):
    is_chemical = get_setting(args.dataset) == 'chemical'
    max_degree = 6 if is_chemical else args.max_degree

    gradient_l1, gradient_l2, _ = get_relevant_gradients(gradients, model)
    feat_dim = gradient_l1.shape[1]
    
    if args.debug:
        gt_fms, _, _, _ = batch
        gt_fms = normalize_features(args, gt_fms)
        gt_fms.requires_grad_(True)
        verify_gt_filter(args, gt_fms, passed_nodes)
    elif batch is not None:
        raise ValueError('Ground truth should not be given if not in debug mode')
        
    if args.use_degree: zero_degrees = passed_nodes[(get_degree(args, passed_nodes.unsqueeze(0)) == 0).squeeze(0)]
    else: zero_degrees = passed_nodes

    for single_X in zero_degrees:
        A = torch.tensor([1]).unsqueeze(0).cuda()
        if scorer.compare_gradients(normalize_adjacency(args, model_args, A), single_X.unsqueeze(0)) < 1e-6:
            extra_rows = torch.zeros(max_degree*max_degree, feat_dim)
            return False, torch.cat((single_X.unsqueeze(0), extra_rows.cuda()), dim=0)
    
    if args.use_degree: list_degrees = list(set(get_degree(args, passed_nodes.unsqueeze(0))[0].tolist()))
    else: list_degrees = range(1, 10)

    def batched_combinations(indices, r, batch_size):
        all_combinations = torch.combinations(indices, r=r, with_replacement=True)
        for i in range(0, all_combinations.size(0), batch_size):
            yield all_combinations[i: i + batch_size]
    
    all_passed_deg1_spancheck = 0    
    list_passed_deg1s = []
    global survived 
    survived = True
    
    start_time = time.time()
    time_limit = 600
    
    timeout = False
            
    with torch.no_grad():
        for i in range(0,max_degree+1):
            
            if i not in list_degrees or passed_nodes.shape[0]**i>10**8: 
                continue

            combinations_gen = batched_combinations(torch.arange(passed_nodes.size(0)), r=i, batch_size=1000)

            if timeout:
                break
            
            for batch_indices in combinations_gen:

                if time.time() > start_time + time_limit:
                    print('Timeout during 1-hop bb generation')
                    timeout = True
                    break
                
                if args.use_degree: valid_degrees = (get_degree(args, passed_nodes.unsqueeze(0))==i)            
                else: valid_degrees = torch.ones(passed_nodes.shape[0], dtype=torch.bool).unsqueeze(0).cuda()

                selected_nodes = passed_nodes[batch_indices]

                current_nodes = passed_nodes[valid_degrees.squeeze(0)].unsqueeze(1)
                current_nodes_repeated = current_nodes.repeat(selected_nodes.shape[0],1,1)
                selected_nodes_repeated = selected_nodes.repeat_interleave(current_nodes.shape[0], dim=0)
                candidate_deg1 = torch.cat((current_nodes_repeated, selected_nodes_repeated), dim=1)

                # Apply span check filtering
                candidate_embeddings = get_l2_embeddings(args, model, candidate_deg1)
                
                _, R_K_norm_GCN_2 = get_layer_decomp(gradient_l2, tol=tol2)
                R_K_norm_GCN_2 = R_K_norm_GCN_2.cuda()

                diffs = check_if_in_span(R_K_norm_GCN_2, candidate_embeddings)

                diffs = torch.log10(diffs)
                diffs = torch.where(diffs == -float('inf'), torch.tensor(-7.0), diffs)
                all_passed_deg1_spancheck += torch.sum(diffs<-2)

                candidate_deg1 = candidate_deg1[diffs<-2]
                zero_rows = torch.zeros(candidate_deg1.shape[0], max_degree*max_degree+1-candidate_deg1.shape[1], feat_dim).cuda()                
                candidate_deg1 = torch.cat((candidate_deg1, zero_rows), dim=1)

                del zero_rows

                if candidate_deg1.shape[0]!=0:
                    list_passed_deg1s.append(properly_structure_deg1(args, candidate_deg1, max_degree, feat_dim))
                torch.cuda.empty_cache()
            
    print('Just generated and filtered all deg1s')     
    
    final_deg1 = torch.cat(list_passed_deg1s)

    del list_passed_deg1s
    
    print('Just going out of bbs_deg1 function')    
    
    if args.use_degree:
        degrees = get_degree(args, final_deg1)[:, 1:]
    
    if not args.directed and args.use_degree: indices_0_dangs = torch.nonzero(torch.all(degrees <= 1, dim=1)).squeeze()
    elif args.directed: indices_0_dangs = torch.nonzero(torch.all(degrees <= 0, dim=1)).squeeze()
        # In the directed case it is worth checking only the bbs where all but the center have in-degree 0
        # We cannot discard afterwards them though
    else: indices_0_dangs = torch.arange(0, final_deg1.shape[0])
        
    if indices_0_dangs.numel()==1: indices_0_dangs = [indices_0_dangs.tolist()]
    else: indices_0_dangs = indices_0_dangs.tolist()

    for ind in indices_0_dangs:
        
        if not args.directed:
            size = torch.count_nonzero(final_deg1[ind][:,0]!=0)
            X = final_deg1[ind][:size]
            A = torch.eye(size)
            A[:,0] = 1
            A[0,:] = 1
        
        if args.directed:
            bb = final_deg1[ind]
            size = (bb.abs().sum(dim=1) != 0).sum().item()
            X = final_deg1[ind][:size]
            A = torch.eye(size)
            A[0,:] = 1
                
        if scorer.compare_gradients(normalize_adjacency(args, model_args, A), X) < 1e-6:              
            return False, final_deg1[ind]   
            
    return True, final_deg1
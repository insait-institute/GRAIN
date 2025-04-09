import torch
from utils import get_layer_decomp, check_if_in_span, possible_feature_values, get_setting
from utils import normalize_features, denormalize_features

def filter_feature_by_feature(args, gradient_GCN_1, tol1):
    setting = get_setting(args.dataset)
    ls_lengths = [len(x) for x in possible_feature_values(args)]
    
    
    if setting == 'citations':
        ls_lengths = [1 if x != 100 else x for x in ls_lengths] # Only the degree feature is one-hot encoded
        # The rest are binary except the extra test ones which are deliberately not one-hot encoded
    for _ in range(args.num_extra_features):
        ls_lengths = [args.max_extra_value] + ls_lengths
    
    passed_nodes = torch.tensor([[]]).cuda()
    ind = 0

    if setting == 'chemical' or args.dataset == 'pokec':
        max_size = 10000
        
    if args.dataset == 'citeseer':
        max_size = 3000

    num_feats = len(ls_lengths)
    
    idx = num_feats-1

    while idx>=0:
        
        ind+=1

        curr_passed = passed_nodes.shape[0]
        potential = curr_passed
        to_end = idx
        f = []

        flag = False

        if idx>=args.num_extra_features:
            for j in range(idx, args.num_extra_features-1, -1):

                if ls_lengths[j]==1:
                    potential*=2
                else:
                    potential*=ls_lengths[j]
                f.append(ls_lengths[j])
                if potential>=max_size or j==args.num_extra_features:
                    to_end = j
                    break   
        else:
            f.append(args.max_extra_value)
            flag = True

        if args.debug:
            print(f'Next starts from {idx}]')
        idx = to_end-1
        
        if potential>2*10**6:
            return None

        tensor_list = [
            torch.tensor([1., 0.]).T.unsqueeze(1).cuda() if x == 1 
            else torch.eye(x).cuda()
            for x in f[::-1]
        ]
        index_grids = torch.meshgrid(*[torch.arange(t.size(0)) for t in tensor_list], indexing="ij")

        f_vec = torch.stack([
            torch.cat([tensor_list[i][index_grids[i].flatten()[j]] for i in range(len(tensor_list))])
            for j in range(index_grids[0].numel())
        ])

        
        tensor_a_expanded = passed_nodes.unsqueeze(1).expand(passed_nodes.shape[0], f_vec.shape[0], passed_nodes.shape[1]) # Shape: (7, 12, 100)
        tensor_b_expanded = f_vec.unsqueeze(0).expand(passed_nodes.shape[0], f_vec.shape[0], f_vec.shape[1])
        combined = torch.cat((tensor_b_expanded,tensor_a_expanded), dim=2) 
        result = combined.reshape(-1, f_vec.shape[1]+passed_nodes.shape[1]).cuda()            
        result = normalize_features(args, result, reverse=True)
        
        ind_f_copy = torch.clone(result)
        _, R_K_norm_l1 = get_layer_decomp(gradient_GCN_1[:, -result.shape[1]:], tol=tol1)
        R_K_norm_l1 = R_K_norm_l1.cuda()

        if args.debug:
            print(f'{passed_nodes.shape[0]} -> {ind_f_copy.shape} -> {idx}')

        diffs = check_if_in_span(R_K_norm_l1, ind_f_copy)
        diffs = torch.log10(diffs)
        diffs = torch.where(diffs == -float('inf'), torch.tensor(-7.0), diffs)
        if not flag:
            passed_nodes = result[diffs<-3]
            passed_nodes = torch.round(denormalize_features(args, passed_nodes, reverse = True))
        else:
            num_ranges = diffs.numel() // args.max_extra_value  
            reshaped_tensor = diffs[:num_ranges * args.max_extra_value].reshape(num_ranges, args.max_extra_value)
            min_indices_within_ranges = torch.argmin(reshaped_tensor, dim=1)
            global_min_indices = min_indices_within_ranges + torch.arange(num_ranges).cuda() * args.max_extra_value
            passed_nodes = result[global_min_indices]
            passed_diffs = diffs[global_min_indices]
            passed_nodes = passed_nodes[passed_diffs<-3]
            passed_nodes = torch.round(denormalize_features(passed_nodes, args.dataset, reverse = True))
            
        
        if passed_nodes.numel()==0:
            print('Could not recover any node')
            return None

    return normalize_features(args, passed_nodes)

def filter_nodes(args, gradient_GCN_1, tol1, f_by_f=True):
    
    print('Filtering started')

    if not f_by_f:
        raise ValueError('We should not be doing full exploration filtering anymore')
    else:
        passed = filter_feature_by_feature(args, gradient_GCN_1, tol1)
        return passed
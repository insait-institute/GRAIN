import torch
import itertools
import matplotlib.pyplot as plt
import neptune
import numpy as np
import io
import math
import statistics
from models import GCN, GAT
from utils import normalize_adjacency, normalize_features
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from utils.constants import POKEC_FEATURES, CHEMICAL_FEATURES


def match_graphs(A1, X1, A2, X2, model):
    assert (hasattr(model, 'get_features') and callable(model.get_features)), "Model should have a method called 'get_features'."
    d = min(A1.shape[0], A2.shape[0])
    
    D_features = ((X1.unsqueeze(0) - X2.unsqueeze(1))**2)
    D_features = D_features.sum(-1)/A1.shape[1]
    
    F1 = model.get_features(A1, X1)
    F2 = model.get_features(A2, X2)
        
    for f1, f2 in zip(F1, F2):
        f1 = torch.where(f1.isnan(), 0, f1)
        f2 = torch.where(f2.isnan(), 0, f2)
        D_features += ((f1.unsqueeze(0) - f2.unsqueeze(1))**2).sum(-1)/A1.shape[1]
    
    D = D_features
    
    row_ind, col_ind = linear_sum_assignment(D.detach().cpu().numpy())
    
    return row_ind, col_ind


def possible_feature_values(args):
    setting = get_setting(args.dataset)
    
    if setting == 'chemical':
        feature_list = deepcopy(CHEMICAL_FEATURES)
    
    elif args.dataset == 'citeseer':
        feature_list = []
        for _ in range(args.num_extra_features):
            feature_list.append(list(range(args.max_extra_value)))
        for _ in range(3703):
            feature_list.append([0.0, 1.0])
        if args.use_degree: feature_list.append(list(range(100)))
    
    elif args.dataset == 'pokec':
        feature_list = deepcopy(POKEC_FEATURES)
        if args.use_degree: feature_list.append(list(range(100)))
        
    return feature_list

def get_edges_37(bb, max_degree):
    list = []
    for i in range(1, max_degree+1):
        if torch.any(bb[0]!=0) and torch.any(bb[i]!=0):
            list.append((0,i))
    for i in range(1, max_degree+1):
        for j in range(2+i*(max_degree-1), 2+(i+1)*(max_degree-1)):
            if torch.any(bb[i]!=0) and torch.any(bb[j]!=0):
                list.append((i, j))
    return list
    
def get_degree(args, x):

    x = torch.tensor(x)
    setting = get_setting(args.dataset)
    if setting == 'chemical':
        while len(x.shape) < 3:
            x = x.unsqueeze(0)
        x = x[ :, :, 58:65]
        v, i = x.max( axis=2 )
        i[ v.isnan() ] = -1
        return i
    
    if args.dataset == 'citeseer':
        while len(x.shape) < 3:
            x = x.unsqueeze(0)
        x = x[ :, :, args.num_extra_features+3703:args.num_extra_features+3803]
        v, i = x.max( axis=2 )
        i[ v.isnan() ] = -1
        i[ v==0 ] = -1
        return i
    
    if args.dataset == 'pokec':
        while len(x.shape) < 3:
            x = x.unsqueeze(0)
        x = x[ :, :, -100:]
        v, i = x.max( axis=2 )
        i[ v.isnan() ] = -1
        i[ v==0 ] = -1
        return i

def properly_structure_deg1(args, deg1s, max_degree, feat_dim): 
        
    centers = deg1s[:, 0, :]
    neighbors = deg1s[:, 1:, :]
    
    if args.use_degree: degrees = get_degree(args, neighbors)
    else: degrees = torch.where(torch.all(neighbors == 0, dim=-1), -1, 78)

    to_sort = torch.cat((degrees.unsqueeze(2), neighbors), dim=-1)
    
    list = []
    
    for i in range(deg1s.shape[0]):
        rows_as_lists = to_sort[i].tolist()
        sorted_indices = sorted(range(len(rows_as_lists)), key=lambda i: rows_as_lists[i], reverse=True)
        sorted_tensor = to_sort[i][sorted_indices]
        list.append(sorted_tensor[:, 1:])
    
    all_sorted = torch.stack(list)
    
    if all_sorted.shape[1]!=max_degree*max_degree or all_sorted.shape[2]!=feat_dim:
        breakpoint()

    assert(all_sorted.shape[1]==max_degree*max_degree)
    assert(all_sorted.shape[2]==feat_dim)    
    final = torch.cat((centers.unsqueeze(1), all_sorted), dim = 1)

    return final

def properly_structure_deg2(args, deg2s, max_degree, feat_dim):
        
    centers = deg2s[:, 0, :]
    centers = centers.cuda()
    next = deg2s[:, 1:, :]

    if args.use_degree: degrees = get_degree(args, next)
    else: degrees = torch.where(torch.all(next == 0, dim=-1), -1, 78)

    to_sort = torch.cat((degrees.unsqueeze(2), next), dim=-1)
    list = []
    
    new37 = max_degree*max_degree+1
        
    for i in range(deg2s.shape[0]):
                
        rows_as_lists = to_sort[i].tolist()

        second_part = []
        index = max_degree
        while index+max_degree<=new37:
            second_part += [rows_as_lists[index:index+max_degree-1]]
            index+=max_degree-1
        
        list_sorted_stuff = []
        for t in second_part:
            sorted_indices = sorted(range(len(t)), key=lambda i: t[i], reverse=True)
            t = [t[j] for j in sorted_indices]
            list_sorted_stuff += [t]
        
        second_part = list_sorted_stuff
                
        ######################################  
        new_rows_as_lists = rows_as_lists[0:max_degree] + [t for x in second_part for t in x]
        
        part0 = []
        index = max_degree
        while index+max_degree<=new37:
            small = (index-1)/(max_degree-1)-1
            # print(small)
            small = int(small)
            part0.append([new_rows_as_lists[small]] + new_rows_as_lists[index:index+max_degree-1])
            # print(f'{small} ; {index}:{index+max_degree-1}')
            index+=max_degree-1
        
        sorted_indices0 = sorted(range(len(part0)), key=lambda i: part0[i], reverse=True)
        
        part0 = [part0[j][0] for j in sorted_indices0]

        second_part = [second_part[j] for j in sorted_indices0]        
                
        final = part0 + [t for x in second_part for t in x]
        
        sorted_tensor = torch.tensor(final)
        list.append(sorted_tensor[:, 1:])
        
    all_sorted = torch.stack(list).cuda()
    assert(all_sorted.shape[1]==max_degree*max_degree)
    assert(all_sorted.shape[2]==feat_dim)    
    final = torch.cat((centers.unsqueeze(1), all_sorted), dim = 1)

    return final


def get_gt_37_deg1(gt_fms, gt_ams, max_degree, feat_dim):
    nn = gt_ams.shape[0]
    list = []
    deg1_size = max_degree*max_degree+1

    for i in range(nn):
        
        curr = [gt_fms[i]] + [gt_fms[x] for x in range(nn) if gt_ams[i][x]==1 and x!=i]
        
        curr = torch.stack(curr)
        
        more = deg1_size-curr.shape[0]
        nans = torch.stack([torch.zeros(feat_dim).cuda() for _ in range(more)])
        curr = torch.cat((curr,nans), dim=0)
        
        assert(curr.shape[0]==deg1_size)
        list.append(curr)
    
    result = properly_structure_deg1(torch.stack(list, dim=0), max_degree, feat_dim)
    
    return result

def get_gt_37_deg2(gt_fms, gt_ams, max_degree, feat_dim):
    nn = gt_ams.shape[0]
    list = []
    
    for i in range(nn):
        
        list_rows = [gt_fms[i]]
        neighbor_indices = [x for x in range(nn) if gt_ams[i][x] and x!=i]
        neighbors_ftm = gt_fms[neighbor_indices]
        
        neighbors_ftm = [row for row in neighbors_ftm]
        
        more = max_degree - len(neighbor_indices)
        nans = [torch.zeros(feat_dim).cuda() for _ in range(more)]
        list_rows += neighbors_ftm
        list_rows += nans
        for j in neighbor_indices:
            two_hop_neighbors_indices = [x for x in range(nn) if gt_ams[j][x] and x!=j and x!=i]
            two_hop_neighbors_ftm = gt_fms[two_hop_neighbors_indices]
            two_hop_neighbors_ftm = [row for row in two_hop_neighbors_ftm]
            more = max_degree-1 - len(two_hop_neighbors_indices)
            nans = [torch.zeros(feat_dim).cuda() for _ in range(more)]
            list_rows += two_hop_neighbors_ftm
            list_rows += nans
        more = max_degree*max_degree+1-len(list_rows)
        nans = [torch.zeros(feat_dim).cuda() for _ in range(more)]
        list_rows += nans
        
        deg2_bb_tensor = torch.stack(list_rows)
        list.append(deg2_bb_tensor) 
    
    result = properly_structure_deg2(torch.stack(list, dim=0), max_degree, feat_dim)
    # breakpoint()
    
    return result

def get_AX_deg1(args, batched_bbs): 
    degrees = get_degree(args, batched_bbs)
    if not args.directed:
        center_degrees = torch.pow(degrees[:,0]+ 1, -0.5)
        center_degrees = torch.where(center_degrees==1, torch.tensor(0.0), center_degrees)
        center_degrees = torch.where(torch.isinf(center_degrees), torch.tensor(0.0), center_degrees).unsqueeze(1) 
        all_degrees = torch.pow(degrees+ 1, -0.5)
        all_degrees = torch.where(all_degrees==1, torch.tensor(0.0), all_degrees)     
        all_degrees = torch.where(torch.isinf(all_degrees), torch.tensor(0.0), all_degrees) 
        As_first_rows = center_degrees * all_degrees
        AXs = torch.bmm(As_first_rows.unsqueeze(1), batched_bbs)
        return AXs
    else:
        center_degrees = torch.pow(degrees[:,0]+ 1, -1.0)
        center_degrees = torch.where(center_degrees==1, torch.tensor(0.0), center_degrees)
        center_degrees = torch.where(torch.isinf(center_degrees), torch.tensor(0.0), center_degrees).unsqueeze(1)
        center_repeated = center_degrees.repeat(1, batched_bbs.shape[1])
        AXs = torch.bmm(center_repeated.unsqueeze(1), batched_bbs)
        return AXs

def get_AX_deg2(args, batched_bbs, gcn_model, max_degree): 
    
    new37 = max_degree*max_degree+1
    new5 = max_degree-1
    if not args.directed:
        degrees = get_degree(args, batched_bbs) + 1
        degrees = torch.where(degrees==1, torch.tensor(0.0), degrees)
        result = torch.bmm(degrees.unsqueeze(2), degrees.unsqueeze(1))
        result = torch.pow(result, -0.5)
        result = torch.where(torch.isinf(result), torch.tensor(0.0), result)
        universal_adj_37 = torch.zeros((new37,new37)).cuda()
        for i in range(max_degree+1):
            universal_adj_37[0,i] = 1
            universal_adj_37[i,0] = 1
        for i in range(1,max_degree+1):
            universal_adj_37[i,new5*i+2:new5*i+max_degree+1] = 1
            universal_adj_37[new5*i+2:new5*i+max_degree+1,i] = 1
        for i in range(new37):
            universal_adj_37[i,i] = 1
        all_adj = result * universal_adj_37
    else:
        
        degrees = get_degree(batched_bbs) + 1
        degrees = torch.where(degrees==1, torch.tensor(0.0), degrees)
        
        result = degrees
        result = torch.pow(result, -1.0)
        result = torch.where(torch.isinf(result), torch.tensor(0.0), result)
        result = result.unsqueeze(2).repeat(1, 1, new37)

        universal_adj_37 = torch.zeros((new37,new37)).cuda()
        for i in range(max_degree+1):
            universal_adj_37[0,i] = 1
        for i in range(1,max_degree+1):
            universal_adj_37[i,new5*i+2:new5*i+max_degree+1] = 1
        for i in range(new37):
            universal_adj_37[i,i] = 1
        all_adj = result * universal_adj_37
        for i in range(new37):
            for j in range(all_adj.shape[0]):
                if all_adj[j,i,i]==0:
                    all_adj[j,i,i] = 1
                    
    AXs = torch.bmm(all_adj, batched_bbs)
    output = all_adj@gcn_model.act(AXs@gcn_model.W_list[0])@gcn_model.W_list[1]
    # AAXs = torch.bmm(all_adj, AXs)
    return output


def are_compatible_pos(args, xs, xs_size, ys, ys_size, pos):
    nn_xs = xs_size[:,None] - 1
    nn_ys = ys_size[:,None] - 1
    if args.use_degree:
        d_xs = get_degree(args, xs[:,1:])
        d_ys = get_degree(args, ys[:,1:])
    
    batch_size = xs.shape[0]

    if args.use_degree:
        can_glue_ys = torch.any( torch.logical_and( torch.all( xs[:,0:1] == ys[:,1:], dim=2 ), d_ys == nn_xs ), axis=1 )
        can_glue_xs = torch.any( torch.logical_and( torch.all( ys[:,0:1] == xs[:,pos:pos+1], dim=2 ), d_xs == nn_ys ), axis=1 )
    else:
        can_glue_ys = torch.any(  torch.all( xs[:,0:1] == ys[:,1:], dim=2 ) , axis=1 )
        can_glue_xs = torch.any( torch.all( ys[:,0:1] == xs[:,pos:pos+1], dim=2 ), axis=1 )
    
    can_glue = torch.logical_and( can_glue_xs, can_glue_ys )
    
    if args.use_degree:
        indices = ( torch.argmax(
                        torch.logical_and( 
                              torch.all( xs[:,0:1] == ys[:,1:], dim=2 ), 
                              d_ys == nn_xs 
                        ).to(dtype=torch.int64),dim=1
                    ) + 
                   torch.ones((batch_size), dtype=torch.int64).cuda() )* can_glue.int()
    else:
        indices = ( torch.argmax(
                        torch.all( xs[:,0:1] == ys[:,1:], dim=2 ).to(dtype=torch.int64),dim=1
                    ) + 
                   torch.ones((batch_size), dtype=torch.int64).cuda() )* can_glue.int()
    return indices

def revert_one_hot_encoding_multiple(args, X):

    lists = sum(possible_feature_values(args), [])
    list_x_reverted = []
    assert(len(lists)==140)
    list_alls = []
    
    for i in range(X.shape[0]):
        x = X[i]
        list_x_reverted = []
        if x.shape[0]==0:
            continue
        for ind in range(140):
            if x[ind]>0:
                list_x_reverted.append(lists[ind])
        list_alls.append(list_x_reverted)
        
    return tuple(list_alls)

def features_combined_ultimate(xs_idxs, ys_idxs, X_cuda_filtered, list_tensors_pos, max_degree, feat_dim):
        
    batch_size = ys_idxs.shape[0]
    num_dangling_bits = ys_idxs.shape[1]
    
    xs = X_cuda_filtered[xs_idxs]
    ys = X_cuda_filtered[ys_idxs]
    
    expanded_xs_idxs = xs_idxs.repeat(1, num_dangling_bits)
    expanded_xs_idxs = expanded_xs_idxs.cuda()
    dang_idx = torch.arange(1, num_dangling_bits + 1).repeat(batch_size, 1)
    dang_idx = dang_idx.cuda()
    indexing = torch.stack((dang_idx, expanded_xs_idxs, ys_idxs), dim=0).cuda()
    indexing = indexing.permute(1, 2, 0).reshape(-1, 3)
    
    db = indexing[:,0]
    x_coord = indexing[:,1]
    y_coord = indexing[:,2]
    
    where_append = list_tensors_pos[db, x_coord, y_coord]
    where_append = where_append.view(batch_size, num_dangling_bits)
    
    
    ys_idx = torch.ones( (batch_size, num_dangling_bits, max_degree+1), dtype=torch.bool, device='cuda' )
    ys_idx[:,:,0] = False

    ys_idx[torch.arange(batch_size).unsqueeze(1), torch.arange(num_dangling_bits).unsqueeze(0), where_append] = False
    ys = ys[:, :, :max_degree+1]
    ys = ys[ys_idx]
    ys=ys.reshape(batch_size,num_dangling_bits*(max_degree-1),feat_dim)
    xs = xs.reshape(batch_size, max_degree*max_degree+1, feat_dim)
    xs = xs[:, :max_degree+1]
    xs=torch.cat((xs, ys), dim=1)
    
    nones = torch.zeros((batch_size, max_degree*max_degree+1-xs.shape[1], feat_dim)).cuda() * torch.nan
    xs=torch.cat((xs, nones), dim=1)
    
    
    return xs



def plot_bb_filter(correct, wrong, use_neptune, name, run):
    plt.clf()
    plt.scatter(range(len(wrong)), wrong, color='blue', marker='x', label='wrong')
    plt.scatter(range(len(correct)), correct, color='red', marker='x', label='correct')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    if use_neptune:
        run[f"charts/{name}"].append(neptune.types.File.from_content(buf.getvalue(), extension="png"))
        
def get_layer_decomp(grad, B=None, tol=None):
    if isinstance(grad, torch.Tensor):
        # If grad is a PyTorch tensor, apply the operations
        grad = grad.detach().cpu().numpy()
    else:
        # Handle other types of objects gracefully
        raise TypeError("Expected grad to be a torch.Tensor, got {} instead.".format(type(grad)))
    
    if B == None:
        B = np.linalg.matrix_rank( grad , tol=tol)
    U,S,Vh = torch.svd_lowrank(torch.tensor(grad),q=B,niter=10)
    R = Vh.T
    return  B, torch.Tensor(R).detach()

def check_if_in_span(R_K_norm, v):
    v /= v.pow(2).sum(-1,keepdim=True).sqrt()
    
    proj = torch.einsum('ik,ij,...j->...k', R_K_norm, R_K_norm, v ) # ( (R_K_norm @ v.T) [:,:,None] * R_K_norm[:,None,:] ).sum(0)
    
    out_of_span = proj - v
    size = out_of_span.abs().sum(-1) # Using L1 norm

    return size

def get_best_hyperparameter(list, biggest_yes):
    mean = statistics.mean(list)
    std_dev = statistics.stdev(list)
    if std_dev!=0:
        hype = (mean - biggest_yes)/std_dev
        return hype
    else:
        print(list)
        print(biggest_yes)
        assert(2<1)
    
def l2_dist(adv_grad, true_grad):
    l2_distances = []

    for tensor1, tensor2 in zip(adv_grad, true_grad):
        squared_diff = (tensor1 - tensor2) ** 2
        sum_squared_diff = squared_diff.sum()
        l2_distance = torch.sqrt(sum_squared_diff)
        l2_distances.append(l2_distance.item())

    mean_l2_distance = torch.tensor(l2_distances).mean()
    return mean_l2_distance.item()

def log_metrics(metrics, run):
    for metric in metrics:
        if metric == 'num_edge_frac' and math.isnan(metrics[metric]):
            run[f"metrics/{metric}"].append(1)
            continue    
        run[f"metrics/{metric}"].append(metrics[metric])
        
def compute_grads_fed_avg(model, criterion, Xs, As, gt_ls, avg_epochs, avg_lr):
    with torch.enable_grad():
        og_weights = [param.data.clone() for param in model.parameters()]

        model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=avg_lr)

        for _ in range(avg_epochs):
            optimizer.zero_grad()
            logits = model(As, Xs)
            loss = criterion(logits, gt_ls).mean()
            loss.backward()
            optimizer.step()
           
        grad = [-(param.data.detach() - og_weights[i])/avg_lr/avg_epochs for i, param in enumerate(model.parameters())]
        
        for i, param in enumerate(model.parameters()):
            param.data = og_weights[i]
            
        return grad

def adj_from_structured_deg2(deg2_bb, max_degree):
    assert(deg2_bb.shape[0] == max_degree**2 + 1), "Maximum degree and building block shape do not match."
    
    A = torch.eye(deg2_bb.shape[0]).cuda()
    A[0, 1:max_degree+1] = 1.
    A[1:max_degree+1, 0] = 1.

    for i in range(1, max_degree+1):
        A[i, i*(max_degree-1)+2:(i+1)*(max_degree-1)+2] = 1.
        A[i*(max_degree-1)+2:(i+1)*(max_degree-1)+2, i] = 1.
    zero_mask = torch.where((deg2_bb == 0).all(1))[0]
    A[zero_mask, :] = 0.
    A[:, zero_mask] = 0.

    return A

def compute_gradients(args, model_args, model, criterion, batch):
    gt_fms, gt_ams, gt_ls, _ = batch
    gt_fms = normalize_features(args, gt_fms)
    gt_fms.requires_grad_(True)

    if not model_args['graph_classification']:
        gt_ls = torch.zeros((gt_fms.shape[0], 2))
        true_idxs = torch.randint(0, 2, size = (gt_fms.shape[0],))
        gt_ls[torch.arange(gt_fms.shape[0]), true_idxs] = 1.
        gt_ls = gt_ls.cuda()

    adj_forward = normalize_adjacency(args, model_args, gt_ams)
    
    global gradient
    if args.federated_optimizer == 'FedSGD':
        if isinstance(model, GCN):
            gcn_output = model.gcn(gt_fms, adj_forward)['emb'].cuda()  
            logits = model.readout(gt_fms, gcn_output).cuda()
            gcn_output.requires_grad_(True)
        elif isinstance(model, GAT):
            out = model(adj_forward, gt_fms, output_hidden_states=True)
            logits = out['logits']
        else:
            raise NotImplementedError(f"Model type {type(model)} is not currently supported")

        loss = criterion(logits, gt_ls).cuda()
        loss = loss.sum() / loss.numel()    
        gradients = torch.autograd.grad(loss, model.parameters())
        
    else:
        gradients = compute_grads_fed_avg(model, criterion, gt_fms, adj_forward, gt_ls, avg_epochs=args.epochs, avg_lr=args.lr)

    return gradients

def get_relevant_gradients(gradients, model):
    if isinstance(model, GCN):
        gradient_l1 = gradients[0].t()
        gradient_l2 = gradients[1].t()
        if model.gcn.n_layers == 2:
            gradient_l3 = gradients[2]
        else:
            gradient_l3 = gradients[2].t()
        
    elif isinstance(model, GAT):
        gradient_l1 = gradients[3]
        gradient_l2 = gradients[8]
        if len(model.gat_net) == 2:
            gradient_l3 = gradients[10]
        else:
            gradient_l3 = gradients[13]
    else:
        raise ValueError(f'Model type {type(model)} is not supported')

    return (gradient_l1, gradient_l2, gradient_l3)

def get_setting(dataset):
    if 'zpn' in dataset:
        return 'chemical'
    elif dataset == 'citeseer':
        return 'citations'
    elif dataset == 'pokec':
        return 'social'
    
    raise ValueError(f'Dataset {dataset} is not currently supported')
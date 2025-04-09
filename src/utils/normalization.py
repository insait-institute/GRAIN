import torch
import yaml

from .constants import *

def normalize_adjacency(
        args,
        model_args,
        adj
    ):
    if model_args['model'] == 'gat':
        return adj
    
    row_sums = torch.sum(adj, dim=1)
    D_matrix = torch.diag(row_sums)    
    if not args.directed:
        inv_sqrt_diagonal = 1.0 / torch.sqrt(D_matrix)
        inv_sqrt_diagonal  =torch.where(inv_sqrt_diagonal == float('inf'), torch.tensor(0.0), inv_sqrt_diagonal)
        return inv_sqrt_diagonal@adj.float()@inv_sqrt_diagonal
    else:
        inv_diagonal = 1.0 / D_matrix
        inv_diagonal  =torch.where(inv_diagonal == float('inf'), torch.tensor(0.0), inv_diagonal)
        return inv_diagonal@adj.float()
    
def get_normalization_factors(args):
    
    if args.dataset == 'zpn/clintox':
        means = CLINTOX_MEANS
        std_devs = CLINTOX_STDS
    
    if args.dataset == 'zpn/bbbp':
        means = BBBP_MEANS
        std_devs = BBBP_STDS
        
    if args.dataset == 'zpn/tox21_srp53':
        means = TOX21_MEANS
        std_devs = TOX21_STDS
        
    if args.dataset == 'pokec':
        means = POKEC_MEANS
        std_devs = POKEC_STDS

    if args.dataset == 'citeseer':
        means = CITESEER_MEANS
        std_devs = CITESEER_STDS

        if not args.use_degree:
            means = means[:3703]
            std_devs = std_devs[:3703]

        extra_means = CITESEER_EXTRA_MEANS
        extra_stds = CITESEEER_EXTRA_STDS

        if args.num_extra_features!=0:
            means = torch.cat((extra_means[-args.num_extra_features:],means), dim=0)
            std_devs = torch.cat((extra_stds[-args.num_extra_features:],std_devs), dim=0)

    return means, std_devs

def normalize_features(args, fts, reverse = False):
    
    means, std_devs = get_normalization_factors(args)
    if not reverse: means, std_devs = means.cuda()[:fts.shape[1]], std_devs.cuda()[:fts.shape[1]]
    else: means, std_devs = means.cuda()[-fts.shape[1]:], std_devs.cuda()[-fts.shape[1]:]
    normalized_fts = (fts.cuda() - means) / std_devs
    normalized_fts = torch.nan_to_num(normalized_fts, nan=0.0, posinf=0.0, neginf=0.0)

    return normalized_fts

def denormalize_features(args, normalized_fts, reverse = False):
    
    means, std_devs = get_normalization_factors(args)
    if not reverse: means, std_devs = means.cuda()[:normalized_fts.shape[1]], std_devs.cuda()[:normalized_fts.shape[1]]
    else: means, std_devs = means.cuda()[-normalized_fts.shape[1]:], std_devs.cuda()[-normalized_fts.shape[1]:]
    fts = normalized_fts.cuda()*std_devs + means
    fts = torch.nan_to_num(fts, nan=0.0, posinf=0.0, neginf=0.0)

    return fts

def normalize_features_indexed(args, fts, idxs):
    
    means, std_devs = get_normalization_factors(args)
    means, std_devs = means.cuda()[idxs], std_devs.cuda()[idxs]
    normalized_fts = (fts.cuda() - means) / std_devs
    normalized_fts = torch.nan_to_num(normalized_fts, nan=0.0, posinf=0.0, neginf=0.0)

    return normalized_fts
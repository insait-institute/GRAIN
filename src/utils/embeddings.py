from models import GCN, GAT
from utils.misc import get_AX_deg1, get_AX_deg2, adj_from_structured_deg2, get_setting
import torch

def get_l2_embeddings(args, model, candidate_deg1):
    if isinstance(model, GAT):
        batch_deg = candidate_deg1.shape[1]
        A = torch.eye(batch_deg)
        A[0, :] = 1.
        A[:, 0] = 1.
        adj_norm = torch.where(A == 0, -torch.inf, 0.).cuda()
        embeddings = model.gat_net[0].forward_batched((candidate_deg1, adj_norm.repeat(candidate_deg1.shape[0],1,1)))[0][:, 0]
    
    elif isinstance(model, GCN):
        AXs = get_AX_deg1(args, candidate_deg1).squeeze(1)
        bbs_output = torch.mm(AXs, model.gcn.W_list[0].cuda()).cuda()
        embeddings = model.gcn.act(bbs_output)
    else:
        raise ValueError(f'Model type {type(model)} is not supported')

    return embeddings

def get_l3_embeddings(args, model, candidate_deg2):
    max_degree = args.max_degree if not get_setting(args.dataset) == 'chemical' else 6
    if isinstance(model, GCN):
        if model.gcn.n_layers == 2:
            output_l2 = torch.cat((candidate_deg2[:,0], get_AX_deg2(args, candidate_deg2, model.gcn, max_degree)[:,0]), dim=1)
        else:
            output_l2 = model.gcn.act(get_AX_deg2(args, candidate_deg2, model.gcn, max_degree)[:,0])
    elif isinstance(model, GAT):
        adjs = torch.cat([adj_from_structured_deg2(deg2bb, max_degree).unsqueeze(0) for deg2bb in candidate_deg2])
        adj_norm = torch.where(adjs == 0, -torch.inf, 0.)
        output_l1 = model.gat_net[0].forward_batched((candidate_deg2, adj_norm))
        output_l2 = model.gat_net[1].forward_batched((output_l1[0], output_l1[1]))[0][:, 0]
    else:
        raise ValueError(f'Model type {type(model)} is not supported')

    return output_l2
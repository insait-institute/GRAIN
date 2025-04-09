import torch
from rdkit import Chem
from utils.misc import possible_feature_values
from data_loading import get_setting
import numpy as np

def normalize_ohe_features(args, X, soft=True):
    dataset = args.dataset
    possible_values = possible_feature_values(args)
    ohe_lens = [0] + [1 if len(ls) == 2 and dataset=='citeseer' else len(ls) for ls in possible_values]
    ohe_cutoffs = np.cumsum(ohe_lens)
    normalized_features_idxs = []
    normalized_features_ohe = []
    
    for i in range(len(ohe_lens) - 1):
        if soft:
            normalized_features_idxs.append(torch.softmax(X[:, ohe_cutoffs[i]:ohe_cutoffs[i+1]], dim=1))
        else:
            normalized_features_idxs.append(torch.argmax(X[:, ohe_cutoffs[i]:ohe_cutoffs[i+1]], dim=1).unsqueeze(1))
            _, max_indices = torch.max(X[:, ohe_cutoffs[i]:ohe_cutoffs[i+1]], dim=1)
            X_new = torch.zeros_like(X[:, ohe_cutoffs[i]:ohe_cutoffs[i+1]]).cuda()
            X_new.scatter_(1, max_indices.unsqueeze(1), 1)
            normalized_features_ohe.append(X_new)
            
    if soft:
        return torch.cat(normalized_features_idxs, dim=1)
    else:
        return torch.cat(normalized_features_idxs, dim=1), torch.cat(normalized_features_ohe, dim=1)
    

def step_loss(args, model, criterion, dummy_X, dummy_A, dummy_label, true_grad,optimizer):
    optimizer.zero_grad()
    model.zero_grad()

    X, A, y = preprocess_data(args, dummy_A, dummy_X, dummy_label)
    logits = model(A, X)
    if isinstance(logits, dict):
        logits = logits['logits']
    loss = criterion(logits, y).to(args.device)
    loss = loss.sum() / loss.numel()         
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    
    grad_diff = 0
    for gx, gy in zip(grad, true_grad): 
        grad_diff += ((gx - gy) ** 2).sum()
    grad_diff.backward()
    return grad_diff

def preprocess_data(args, dummy_A, dummy_X, dummy_label):
    dummy_X_norm = normalize_ohe_features(args, dummy_X)
        
    if not args.fix_A:
        dummy_A_triu = dummy_A.triu()
        dummy_A_sym = torch.sigmoid(dummy_A_triu + dummy_A_triu.T)*(1-torch.eye(dummy_A.shape[0]).to(args.device)) + torch.eye(dummy_A.shape[0]).to(args.device)
    else:
        dummy_A_sym = dummy_A
        
    if not args.fix_y:
        dummy_label_norm = torch.softmax(dummy_label, 0)
    else:
        dummy_label_norm = dummy_label
        
    return dummy_X_norm, dummy_A_sym, dummy_label_norm
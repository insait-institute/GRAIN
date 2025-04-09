import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score
from utils.misc import match_graphs, possible_feature_values, get_setting

import yaml

def ohe_accuracy(args, X_true, X_pred):    
    setting = get_setting(args.dataset)
    if setting == 'chemical' or setting == 'social':
        feature_vals = possible_feature_values(args)
    
    if args.dataset == 'citeseer':
        feature_vals = []
        for _ in range(3703):
            feature_vals.append([0.0])
        if args.use_degree: feature_vals.append(list(range(100)))
    
    total_len = sum([len(f) for f in feature_vals])
        
    accuracies = {'p': 0, 'r': 0, 'f': 0}
    curr = 0
    
    for features in feature_vals:
        
        idx_true = X_true[:, curr:curr+len(features)].detach().argmax(1).cpu().numpy()
        idx_pred = X_pred[:, curr:curr+len(features)].detach().argmax(1).cpu().numpy()
        
        accuracies['p'] += precision_score(idx_true, idx_pred, average='weighted')*len(features)/total_len
        accuracies['f'] += recall_score(idx_true, idx_pred, average='weighted')*len(features)/total_len
        accuracies['r'] += f1_score(idx_true, idx_pred, average='weighted')*len(features)/total_len
        curr += len(features)
    
    return accuracies    
    
def evaluate_metrics(args, A_true, X_true, A_pred, X_pred, model):
    if A_pred is None:
        metrics = {}
        
        metrics['deg_0_p'] = 0.
        metrics['deg_0_r'] = 0.
        metrics['deg_0_f'] = 0.
        metrics['num_edge_frac'] = 0.
        
        F1 = model.get_features(A_true, X_true)
        for i, f1 in enumerate(F1):
            f1_np = f1.detach().flatten().cpu().numpy()
            
            metrics[f'deg_{i+1}_r2'] = 0.
            metrics[f'deg_{i+1}_mse'] = ((f1_np - f1_np.mean())**2).mean()
            metrics[f'deg_{i+1}_mse_norm'] = ((f1_np - f1_np.mean())**2).mean()
            metrics[f'deg_{i+1}_mae'] = abs((f1_np - f1_np.mean())).mean()
            metrics[f'deg_{i+1}_mae_norm'] = abs((f1_np - f1_np.mean())).mean()
            metrics[f'deg_{i+1}_mean'] = ((f1_np - f1_np.mean())**2).mean()
            
            metrics[f'deg_{i+1}_acc_10p'] = 0.
            metrics[f'deg_{i+1}_acc_15p'] = 0.
            metrics[f'deg_{i+1}_acc_25p'] = 0.
            
        return metrics
    assert (hasattr(model, 'get_features') and callable(model.get_features)), "Model should have a method called 'get_features'."
    
    A1_size = A_true.shape[0]
    A2_size = A_pred.shape[0]
    
    factor = min(A1_size, A2_size)/max(A1_size, A2_size)
    
    metrics = {}
        
    n_edge_1 = (A_true.sum() - A1_size)//2
    n_edge_2 = (A_pred.sum() - A2_size)//2
    if n_edge_2 == 0 and n_edge_2 == 0:
        metrics['num_edge_frac'] = 1.
    else:
        metrics['num_edge_frac'] = (min(n_edge_1, n_edge_2)/max(n_edge_1, n_edge_2)).cpu().item()
    
    match_rows, match_cols = match_graphs(A_true-torch.eye(A1_size).to(A_true.device), X_true, 
                                          A_pred-torch.eye(A2_size).to(A_pred.device), X_pred, model)
    
    A_true = A_true[match_cols][:, match_cols]
    X_true = X_true[match_cols]
    
    A_pred = A_pred[match_rows][:, match_rows]
    X_pred = X_pred[match_rows]
    
    if args.do_ohe:
        accuracies = ohe_accuracy(args, X_true, X_pred)
        metrics['deg_0_p'] = accuracies['p']*factor
        metrics['deg_0_r'] = accuracies['r']*factor
        metrics['deg_0_f'] = accuracies['f']*factor
    else:
        metrics['deg_0_p'] = precision_score(X_true.detach().flatten().cpu().numpy(), X_pred.detach().flatten().cpu().numpy(), average='weighted')*factor
        metrics['deg_0_r'] = recall_score(X_true.detach().flatten().cpu().numpy(), X_pred.detach().flatten().cpu().numpy(), average='weighted')*factor
        metrics['deg_0_f'] = f1_score(X_true.detach().flatten().cpu().numpy(), X_pred.detach().flatten().cpu().numpy(), average='weighted')*factor
        
    F1 = model.get_features(A_true, X_true)
    F2 = model.get_features(A_pred, X_pred)
        
    for i, (f1, f2) in enumerate(zip(F1, F2)):
        f1_np, f2_np = f1.detach().flatten().cpu().numpy(), f2.detach().flatten().cpu().numpy()
        
        metrics[f'deg_{i+1}_r2'] = max(r2_score(f1_np, f2_np)*factor, 0)
        metrics[f'deg_{i+1}_mse'] = ((f1_np - f2_np)**2).mean()
        metrics[f'deg_{i+1}_mse_norm'] = ((f1_np - f2_np)**2).mean()*factor**2
        metrics[f'deg_{i+1}_mae'] = abs((f1_np - f2_np)).mean()
        metrics[f'deg_{i+1}_mae_norm'] = abs((f1_np - f2_np)).mean()*factor
        metrics[f'deg_{i+1}_mean'] = ((f1_np - f1_np.mean())**2).mean()
        
        f1_std = f1_np.std()
        metrics[f'deg_{i+1}_acc_10p'] = (((f1_np - 0.13*f1_std < f2_np) & (f2_np < f1_np + 0.13*f1_std)).astype(np.int32)).mean()
        metrics[f'deg_{i+1}_acc_15p'] = (((f1_np - 0.19*f1_std < f2_np) & (f2_np < f1_np + 0.19*f1_std)).astype(np.int32)).mean()
        metrics[f'deg_{i+1}_acc_25p'] = (((f1_np - 0.319*f1_std < f2_np) & (f2_np < f1_np + 0.319*f1_std)).astype(np.int32)).mean()

    
    return metrics
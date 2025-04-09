from baseline.args_factory import get_args
from baseline import step_loss, preprocess_data, normalize_ohe_features
import sys
from utils import match_graphs, evaluate_metrics, draw_atom, get_model, possible_feature_values
import yaml
import neptune.new as neptune
import torch
from rdkit import Chem
import matplotlib.pyplot as plt
import io
import numpy as np
from tqdm import tqdm
from data_loading import get_dataset, get_setting, get_dataset_stats

def run_iteration(args, model, criterion, dummy_X, dummy_A, dummy_label, true_gradient, sample_idx, ens_idx):
    
    to_optim = []
    for component in [dummy_X, dummy_A, dummy_label]:
        if component.requires_grad:
            to_optim.append(component)
            
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(to_optim, lr=0.001)
    elif args.optimizer == 'lbfgs':
        optimizer = torch.optim.LBFGS(to_optim, lr=0.1)
    else:
        raise ValueError(f'Optimizer {args.optimizer} is not supported, please select either "sgd" or "lbfgs".') 
    
        
    # Run attack 
    for i in range(args.max_iter):        
        optimizer.zero_grad()
        step_fn = lambda: step_loss(args, model, criterion, dummy_X, dummy_A, dummy_label, true_gradient, optimizer)
        optimizer.step(step_fn)
        if i % 10 == 0 and args.neptune: 
            loss = step_loss(args, model, criterion, dummy_X, dummy_A, dummy_label, true_gradient, optimizer)
            args.neptune[f'logs/{sample_idx}/loss_{ens_idx}'].log(loss)
            
    return preprocess_data(args, dummy_A, dummy_X, dummy_label)

def match_multiple_graphs(args, Xs, As, model):
    '''
    Reorders the Xs and As based on the best match with the first graph.
    '''
    true_size = As[0].shape[0]
    for A in As:
        assert (A.shape[0] == true_size)
        
    Xs_reordered = [Xs[0].unsqueeze(0)]
    As_reordered = [As[0].unsqueeze(0)]
    
    # Perform per-graph matching with the first element
    for X, A in zip(Xs[1:], As[1:]):
        row_idxs, col_idxs = match_graphs(A, X, As[0], Xs[0], model)
        
        # Make sure the first graph is not reordered
        assert(np.all(row_idxs == np.arange(true_size)))
        Xs_reordered.append(X[col_idxs].unsqueeze(0))
        As_reordered.append(A[col_idxs][:, col_idxs].unsqueeze(0))
        
    return Xs_reordered, As_reordered

def run_attack(args, model, criterion, sample, sample_idx, generator=None):
    gt_fms, gt_ams, gt_ls, _  = sample

    # Compute the correct gradients
    logits = model(gt_ams, gt_fms)
    if isinstance(logits, dict):
        logits = logits['logits']
    loss = criterion(logits, gt_ls).to(args.device)
    loss = loss.sum() / loss.numel()         
    true_gradient = torch.autograd.grad(loss, model.parameters())
    
    recon_Xs = []
    recon_As = []
    
    # Extract the ensemble's results
    for iter in tqdm(range(args.n_ens)):
        
        # Add relevant inputs for optimization
        if not args.fix_A:
            dummy_A = torch.randn(gt_ams.shape, requires_grad=True, generator=generator, device=args.device)
        else:
            dummy_A = gt_ams.clone()
            
        if not args.fix_X:
            dummy_X = torch.randn(gt_fms.shape, requires_grad=True, generator=generator, device=args.device)
        else:
            dummy_X = gt_fms.clone()
            
        if not args.fix_y:
            dummy_label = torch.randn(gt_ls.shape, requires_grad=True, generator=generator, device=args.device)
        else:
            dummy_label = gt_ls.clone()
            
        recon_X, recon_A, _ = run_iteration(args, model, criterion, dummy_X, dummy_A, dummy_label, true_gradient, sample_idx, iter)
        
        recon_Xs.append(recon_X)
        recon_As.append(recon_A)
        
    if not args.fix_A:
        recon_Xs, recon_As = match_multiple_graphs(args, recon_Xs, recon_As, model)
        recon_X = torch.cat(recon_Xs, axis=0).median(0).values
        recon_A = torch.cat(recon_As, axis=0).median(0).values
    else:
        recon_X = torch.cat([X.unsqueeze(0) for X in recon_Xs], axis=0).median(0).values
        recon_A = torch.cat([A.unsqueeze(0) for A in recon_As], axis=0).median(0).values

    recon_A = (recon_A > args.A_thrs).int()
    
    return recon_X, recon_A
    
if __name__ == '__main__':
    args, run = get_args()
    
    # Load dataset
    dataset = get_dataset(args)

    
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
          
    model_args = config['model_args']
    
    if args.neptune:
        for arg in model_args:
            run[f'parameters/model_{arg}'] = model_args[arg]

    feat_dim, num_cats = get_dataset_stats(args, args.dataset)
    
    # Load model
    model = get_model(model_args, feat_dim = feat_dim, num_cats = num_cats).to(args.device)
    
    with open(args.eval_config_path, 'r') as file:
        eval_config = yaml.safe_load(file)
          
    eval_model_args = eval_config['model_args']
    eval_model = get_model(eval_model_args, feat_dim = feat_dim, num_cats = num_cats).to(args.device)
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator(device = args.device)
    generator.manual_seed(0) 
    
    
    for i,batch in enumerate(dataset):
        if i == args.n_inputs:
            break
        print(f'Running example {i+1}')
        print('-----------------------')
        
        recon_X, recon_A = run_attack(args, model, criterion, batch, i+1, generator=generator)
        
        if args.do_ohe:
            recon_X_id, recon_X= normalize_ohe_features(args, recon_X, soft=False)
           
        gt_fms, gt_ams, gt_ls, gt_mol = batch

        metrics = evaluate_metrics(args, gt_ams, gt_fms, recon_A, recon_X, eval_model)
        
        
        for metric in metrics:
            print(f'{metric}: {metrics[metric]:.6f}')
            if args.neptune:
                run[f'logs/{metric}'].log(metrics[metric])
                
        # Draw examples
        if 'zpn' in args.dataset:
            os.system('mkdir -p outputs')
            feature_vals = possible_feature_values(args)
            recon_X = torch.tensor([[feature_vals[j][idx.item()] for j, idx in enumerate(recon_X_id[i])] for i in range(recon_X_id.shape[0])]).to(args.device)
            try:
                # Draw examples
                recon = draw_atom(recon_X, recon_A)
                plt.clf()
                plt.imshow(recon)
                
                if args.neptune:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    run[f"images/pred_{i}.png"].append(neptune.types.File.from_content(buf.getvalue(), extension="png"))
                else:
                    plt.savefig(f"outputs/pred_{i}.png")
                    
            except:
                pass
            
            gt = Chem.Draw.MolToImage(gt_mol, size=(1200, 1200))
            buf = io.BytesIO()
            gt.save(buf, format='PNG')
            if args.neptune:
                run[f"images/gt_{i}.png"].append(neptune.types.File.from_content(buf.getvalue(), extension="png"))
            else:
                with open(f"outputs/gt_{i}.png", "wb") as f:
                    f.write(buf.getbuffer())

            


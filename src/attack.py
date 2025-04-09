import torch
import time
import yaml
import math
import neptune
import io
from utils import normalize_adjacency, normalize_features, get_setting, compute_gradients, properly_structure_deg2, get_relevant_gradients, draw_atom, revert_one_hot_encoding_multiple
from graph_reconstruction import filter_bbs, build_molecule_from_bbs_DFS, greedy_building
from rdkit.Chem import Draw
from rdkit import Chem
from utils import get_model, convert_to_bb, SimilarityScorer
from data_loading import get_dataset, get_dataset_stats
from args_factory import get_args
from tabulate import tabulate


def build_graph(args, model_args, scorer, run, filtered_bbs, bb_probs, backup_bbs, backup_bb_probs, upper_bound, max_rank):
    hidden_size = model_args['hidden_size']
    filtered_bbs = convert_to_bb(args, filtered_bbs, max_degree, feat_dim, hidden_size)
    backup_bbs = convert_to_bb(args, backup_bbs, max_degree, feat_dim, hidden_size)

    print(f'------> {len(filtered_bbs[0].A)}')

    building_mode = 'dfs'
    DFS_max_depth = min(max(7, max_rank//3), 20)
    
    if run:
        run["building/building_mode"] = building_mode
        run["building/DFS max depth"] = DFS_max_depth
    
    if building_mode == 'dfs':
        if run:
            run["building/building time limit"] = 900
        accuracy2, pred, depth, dfs_steps = build_molecule_from_bbs_DFS(args, model_args, scorer, feat_dim=feat_dim, DFS_max_depth=DFS_max_depth, bbs=filtered_bbs, backup_bbs = backup_bbs, backup_bb_probs = backup_bb_probs,  deg=2, model = model, criterion = criterion, gt_gradient = gradients, max_degree = max_degree, bbs_prob = bb_probs, upper_bound_atoms=upper_bound, feature_onehot_encoding=True, gt_ams=gt_ams, gt_ls=gt_ls if not model_args['graph_classification'] else None, gt_fms=gt_fms)
        
        if run and pred is not None:
            run["building/successfully_built"].append(accuracy2)
            if dfs_steps is not None:
                run["building/DFS depth"].append(depth)
                run["building/DFS steps"].append(dfs_steps)
            
    return pred

def print_metric(metrics, total_metrics, n_iter):
    to_print = [['Metric', 'Current', 'Aggregate']]
    for metric in metrics:
        if 'r2' not in metric and 'f' not in metric:
            continue
        if metric not in total_metrics:
            total_metrics[metric] = 0
        total_metrics[metric] += metrics[metric]
        to_print.append([metric, f'{metrics[metric]*100:.2f}', f'{total_metrics[metric]/n_iter*100:.2f}'])
        
    print(tabulate(to_print, headers='firstrow', tablefmt='fancy_grid'))
        
def log_metrics(metrics, run):
    for metric in metrics:
        if metric == 'num_edge_frac' and math.isnan(metrics[metric]):
            run[f"metrics/{metric}"].append(1)
            continue    
        run[f"metrics/{metric}"].append(metrics[metric])

def draw_molecule(gt_mol):
    if gt_mol is None:
        return
    img = Draw.MolToImage(gt_mol)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    run[f"images/gt_{i}"].append(neptune.types.File.from_content(img_byte_arr, extension='png'))

def draw_recon_molecule(args, recon_X, recon_A):

    img_reconstruct = draw_atom(revert_one_hot_encoding_multiple(args, recon_X), recon_A)
    img_byte_arr2 = io.BytesIO()
    img_reconstruct.save(img_byte_arr2, format='PNG')
    img_byte_arr2 = img_byte_arr2.getvalue()
    run[f"images/pred_{i}"].append(neptune.types.File.from_content(img_byte_arr2, extension='png'))


def run_attack(args, model_args, run, model, scorer, gradients, criterion, tol1, tol2, tol3):
    build_marker, filtered_bbs, bb_probs, all_bbs, all_bb_probs, filtered_deg1s = filter_bbs(args, model_args, run, model, scorer, criterion, gradients, tol1, tol2, tol3, batch = None, is_gc = model_args['graph_classification'])

    if filtered_bbs is None or len(filtered_bbs) == 0:
        return None, None
    else:
        print(f'Filtered {filtered_bbs.shape[0] if len(filtered_bbs.shape) > 2 else 1} 2-hop building blocks')

    
    while filtered_bbs.dim()!=3:
        filtered_bbs = filtered_bbs.unsqueeze(0)
            
    while filtered_deg1s.dim()!=3:
        filtered_deg1s = filtered_deg1s.unsqueeze(0)

    if not args.naive_build and not is_chemical: 

        pred = greedy_building(args, model_args, filtered_bbs, feat_dim)
        pred2 = greedy_building(args, model_args, filtered_deg1s, feat_dim)

        pred_A, pred_X = pred
        pred_A_all, pred_X_all = pred2

        dist1 = scorer.compare_gradients(normalize_adjacency(args, model_args, pred_A), pred_X)
        dist2 = scorer.compare_gradients(normalize_adjacency(args, model_args, pred_A_all), pred_X_all)

        print(f'Prediction distance: {dist1} and {dist2}')
        
        if dist2<dist1: pred_A, pred_X = pred_A_all, pred_X_all 

        return pred_A.cuda(), pred_X.cuda()  

    if build_marker is None:     
        return pred_A, pred_X

            
    if build_marker:
        # Try building
        filtered_bbs = properly_structure_deg2(args, filtered_bbs, max_degree, feat_dim)
        gradient_l1, gradient_l2, gradient_l3 = get_relevant_gradients(gradients, model)

        max_rank = max([torch.linalg.matrix_rank(gradient_l1, tol=tol1), torch.linalg.matrix_rank(gradient_l2, tol=tol2), torch.linalg.matrix_rank(gradient_l3, tol=tol3)])

        pred = build_graph(args, model_args, scorer, run, filtered_bbs, bb_probs, all_bbs, all_bb_probs, upper_bound = 3*max_rank, max_rank=max_rank)
        
    else:
        # Otherwise, reconstruct graph from the proposed bb
        if filtered_bbs.dim() == 1:
            filtered_bbs = filtered_bbs.unsqueeze(0)
        
        if filtered_bbs.dim() == 3:
            filtered_bbs = filtered_bbs.squeeze(0)

        nonzero_rows = torch.nonzero( torch.any(filtered_bbs != 0, dim=1) ).squeeze()
        
        bb_size = max_degree*max_degree+1
        
        A = torch.eye(bb_size).cuda()
        A[0,:] = 1
        A[:,0] = 1
        for i in range(1,max_degree+1):
            A[i,(max_degree-1)*i+2:(max_degree-1)*i+max_degree+1] = 1
            if not args.directed:
                A[(max_degree-1)*i+2:(max_degree-1)*i+max_degree+1,i] = 1
        
        X = filtered_bbs[nonzero_rows]
        
        A = A[nonzero_rows][:, nonzero_rows]
                    
        pred_A = (A, X)
            
    if pred is None:
        num_deg2 = filtered_bbs.shape[0]
        rank2 = torch.linalg.matrix_rank(gradient_l2, tol=tol2)

        if num_deg2>=2/3*rank2:
            pred = greedy_building(args, model_args, filtered_bbs, feat_dim)
        else:
            pred = greedy_building(args, model_args, filtered_deg1s, feat_dim)

        return torch.tensor(pred[0]).cuda(), torch.tensor(pred[1]).cuda() 
    
    pred_A = torch.tensor(pred[0]).cuda()
    pred_X = torch.tensor(pred[1]).cuda()
    
    if pred_A.dim() == 0:
        pred_A = pred_A.unsqueeze(0).unsqueeze(0)
        pred_X = pred_X.unsqueeze(0)

    return pred_A, pred_X

if __name__ == '__main__':
        

    args, run = get_args()

    if not args.debug:
        import warnings
        warnings.filterwarnings("ignore")

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_args = config['model_args']
    model_type = model_args['model']

    tol1 = config['attack_args']['tol_1']
    tol2 = config['attack_args']['tol_2']
    tol3 = config['attack_args']['tol_3']

    with open(args.eval_config_path, 'r') as file:
        config = yaml.safe_load(file)
            
    model_args_eval = config['model_args']
    
    if run:
        run["model_info/hidden size"] = model_args['hidden_size']
        run["model_info/node emb dim"] = model_args['node_embedding_dim']
        run["model_info/readout hidden dim"] = model_args['readout_hidden_dim']
        run["model_info/graph hidden dim"] = model_args['graph_embedding_dim']
        run["max_nodes_threshold"] = args.min_nodes
        run["min_nodes_threshold"] = args.max_nodes
        
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        
    feat_dim, num_cats = get_dataset_stats(args, args.dataset)
    
    # Load model
    model = get_model(model_args, feat_dim = feat_dim, num_cats = num_cats).to(args.device)
    
    if args.saved_model_path is not None:
        model.load_state_dict(torch.load(args.saved_model_path))
    
    model.to('cuda')
        
    eval_model = get_model(model_args_eval, feat_dim, num_cats)
    eval_model.to('cuda')
    
    dataset = get_dataset(args)
        
    total_graphs = 0
    total_reconstructed = 0
    total_failed = 0
    total_metrics = {}
    
    full_rank_A = 0

    for i, batch in enumerate(dataset):
        gt_fms, gt_ams, gt_ls, gt_mol = batch
        gt_fms = normalize_features(args, gt_fms)
        num_nodes = batch[0].shape[0]
        is_chemical = get_setting(args.dataset) == 'chemical'
        max_degree = args.max_degree if not is_chemical else 6

        if args.dataset in ['pokec', 'citeseer']:
            print(f'Running on graph with {num_nodes} nodes')
            if run:
                run["num nodes"].append(num_nodes)            
                run["rank"].append(torch.linalg.matrix_rank(batch[0]).item()/num_nodes)

        if is_chemical:
            smiles = Chem.MolToSmiles(gt_mol)
            print(i,smiles)    
            if run:
                run["num nodes"].append(num_nodes)            
                run["smiles"].log(smiles)
        
        gradients = compute_gradients(args, model_args, model, criterion, batch)
        scorer = SimilarityScorer(args, model_args, model, eval_model, criterion, gt_X=gt_fms, gt_A=gt_ams, gt_Y=gt_ls if not model_args['graph_classification'] else None, gt_gradients = gradients)

        pred_A, pred_X = run_attack(args, model_args, run, model, scorer, gradients, criterion, tol1, tol2, tol3)

        metrics = scorer.evaluate_similarity(pred_A, pred_X)
            
        if run:
            log_metrics(metrics, run)
        
        print_metric(metrics, total_metrics, n_iter=i+1)

        if is_chemical:               
            if run:
                draw_molecule(gt_mol)
                if pred_A is not None:
                    draw_recon_molecule(args, pred_X, pred_A)


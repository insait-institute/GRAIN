from utils.misc import get_setting, l2_dist, match_graphs
from utils.metrics import evaluate_metrics
import torch
from models import GAT, GCN
from typing import Union

class SimilarityScorer:
    def __init__(self, args, model_args, model: Union[GCN, GAT], eval_model: Union[GCN, GAT], criterion, gt_X:torch.tensor, gt_A:torch.tensor, gt_Y:torch.tensor = None, gt_gradients=None):
        self.args = args
        self.model_args = model_args
        self.model = model
        self.criterion = criterion
        self.eval_model = eval_model
        self.gt_X = gt_X
        self.gt_A = gt_A
        self.gt_Y = gt_Y
        self.gt_gradients = gt_gradients

    def compare_gradients(self, pred_A: torch.tensor, pred_X: torch.tensor) -> float:
        """
        Computes the L2 distance between the model weight gradients produced by the prediction and those produced by the client input.
        """
        setting = get_setting(self.args.dataset)
        assert(self.gt_gradients is not None), 'Cannot do gradient distance comparison when the ground truth gradients have not been given'
        if self.gt_Y is not None and self.model_args['graph_classification']:
            raise ValueError('You are not allowed to give ground truth labels for graph classification')
        
        with torch.enable_grad():
            # Compute model gradients
            if isinstance(self.model, GAT):
                logits_adv = self.model(pred_A.cuda(),pred_X.cuda())['logits']
            elif isinstance(self.model, GCN):
                logits_adv = self.model(pred_A, pred_X)
            else:
                raise ValueError(f'Model type {type(self.model)} is not supported')        
            
            # Try out all label possibilities for the graph classification setting
            if self.model_args['graph_classification']:
                if setting == 'chemical':
                    possible_labels = [torch.tensor([0.,1.]).cuda(),torch.tensor([1.,0.]).cuda()]
                else:
                    possible_labels = [torch.tensor([1.,0.,0.,0.,0.,0.]).cuda(),torch.tensor([0.,1.,0.,0.,0.,0.]).cuda(),torch.tensor([0.,0.,1.,0.,0.,0.]).cuda(),torch.tensor([0.,0.,0.,1.,0.,0.]).cuda(),torch.tensor([0.,0.,0.,0.,1.,0.]).cuda(),torch.tensor([0.,0.,0.,0.,0.,1.]).cuda()]
                    
                l2_diffs = []
                
                for y in possible_labels:
                    
                    adv_loss = self.criterion(logits_adv, y)
                    adv_loss = adv_loss.mean()
                    adv_grad = torch.autograd.grad(adv_loss, self.model.parameters(), create_graph=True)
                    l2_diff = l2_dist(adv_grad, self.gt_gradients)
                    l2_diffs.append(l2_diff)


                    # Debug statement to verify 
                    if self.args.debug:
                        print(f'with {y} -> {l2_diff}')
                                
                return min(l2_diffs)
            else:
                y = torch.zeros((pred_A.shape[0],2)).cuda()
                y[:, 1] = 1.
                with torch.no_grad():
                    row_ids, col_ids = match_graphs(pred_A.cuda(), pred_X.cuda(), self.gt_A.cuda(), self.gt_X.cuda(), self.model)
                    y[col_ids] = self.gt_Y[row_ids]

                adv_loss = self.criterion(logits_adv, y)
                adv_loss = adv_loss.sum() / adv_loss.numel()  
                adv_grad = torch.autograd.grad(adv_loss, self.model.parameters(), create_graph=True)
                l2_diff = l2_dist(adv_grad, self.gt_gradients)
                return l2_diff

    def evaluate_similarity(self, pred_A: torch.tensor, pred_X: torch.tensor):
        """Wrapper around the evaluate_metrics function"""
        return evaluate_metrics(self.args, self.gt_A, self.gt_X, pred_A, pred_X, self.eval_model)
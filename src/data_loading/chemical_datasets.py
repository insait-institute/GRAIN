from data_loading.graph_dataset import GraphDataset
from utils import mol_to_features, mol_to_adj
from datasets import load_dataset
from torch.utils.data import DataLoader
from rdkit import Chem
import torch

class ChemicalDataset(GraphDataset):
    def __init__(self, args):
        assert (args.dataset in ['zpn/bbbp', 'zpn/clintox', 'zpn/tox21_srp53']), \
            "Chemical dataset should be among 'zpn/bbbp', 'zpn/clintox', or 'zpn/tox21_srp53'."
        
        self.X = []
        self.A = []
        self.y = []
        self.mols = []


        dataset = load_dataset(args.dataset)
        dataset = dataset.shuffle(seed=args.rng_seed)
        dataloader = DataLoader(dataset.with_format("torch")['train'], batch_size=1, shuffle=False)

        for i, sample in enumerate(dataloader):
            if i >= args.n_inputs:
                break
            gt_mol = Chem.MolFromSmiles(sample['smiles'][0])
            gt_fms = torch.tensor(mol_to_features(args, gt_mol, feature_onehot_encoding=True)).to(args.device)
            gt_fms.requires_grad_(True)

            gt_ams = torch.tensor(mol_to_adj(gt_mol)).to(args.device)


            label = sample['target']
            if label==1:
                gt_ls = torch.tensor([0.,1.]).to(args.device)
            else:
                gt_ls = torch.tensor([1.,0.]).to(args.device)

            self.X.append(gt_fms)
            self.A.append(gt_ams)
            self.y.append(gt_ls)
            self.mols.append(gt_mol)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.A[idx], self.y[idx], self.mols[idx])
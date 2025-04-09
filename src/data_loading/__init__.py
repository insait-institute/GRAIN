from data_loading.subgraph_dataset import SubgraphDataset
from data_loading.chemical_datasets import ChemicalDataset
from data_loading.data_utils import get_pokec_data, get_citeseer_data, get_setting, get_dataset_stats
import random

def get_dataset(args):
    random.seed(args.rng_seed)
    if args.dataset in ['zpn/bbbp', 'zpn/clintox', 'zpn/tox21_srp53']:
        return ChemicalDataset(args)
    elif args.dataset in ['citeseer', 'pokec']:
        if args.dataset == 'citeseer':
            X, A, y = get_citeseer_data()
            return SubgraphDataset(args, X, A, classes=y)
        else:
            X, A, neighbours = get_pokec_data(args)
            return SubgraphDataset(args, X, A, neighbors_list=neighbours)
    else:
        raise ValueError(f'Dataset {args.dataset} is not currently supported.')
    
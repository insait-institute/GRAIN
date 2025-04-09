import random
import pandas as pd
import torch
import pickle

def get_setting(dataset):
    if 'zpn' in dataset:
        return 'chemical'
    elif dataset == 'citeseer':
        return 'citations'
    elif dataset == 'pokec':
        return 'social'
    
    raise ValueError(f'{dataset} is not a supported dataset')

def get_dataset_stats(args, dataset):
    if 'zpn' in dataset:
        return (140, 2)
    elif dataset == 'citeseer':
        if args.use_degree: 
            feat_dim = 3803
        else: 
            feat_dim = 3703
        feat_dim += args.num_extra_features
        return (feat_dim, 6)
    elif dataset == 'pokec':
        feat_dim = 449 if args.use_degree else 349
        return (feat_dim, 6)
    raise ValueError(f'{dataset} is not a supported dataset')


def get_citeseer_data():
    papers = pd.read_csv('data/citeseer.content', delim_whitespace=True, header=None)

    print(f"Shape of papers DataFrame: {papers.shape}")

    num_words = 3703
    papers.columns = ['paper_id'] + [f'word_{i}' for i in range(1, num_words + 1)] + ['class_label']

    papers['paper_id'] = papers['paper_id'].astype(str)

    file_path = './data/citeseer.content'
    features_list = []
    class_list = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            features = [int(x) for x in parts[1: num_words+1]]
            features_list.append(features)
            class_list += [parts[num_words+1]]
    features_tensor = torch.tensor(features_list, dtype=torch.float32)

    string_to_number = {string: index for index, string in enumerate(set(class_list))}
    class_list = [string_to_number[string] for string in class_list]
    class_tensor = torch.tensor(class_list, dtype=torch.float32)

    edges = pd.read_csv('data/citeseer.cites', delim_whitespace=True, header=None)
    edges.columns = ['cited', 'citing']

    edges['cited'] = edges['cited'].astype(str)
    edges['citing'] = edges['citing'].astype(str)

    id_to_index = {paper_id: index for index, paper_id in enumerate(papers['paper_id'])}

    missing_citing = edges['citing'][~edges['citing'].isin(id_to_index.keys())].unique()
    missing_cited = edges['cited'][~edges['cited'].isin(id_to_index.keys())].unique()

    if missing_citing.size > 0:
        print("Missing citing IDs:", missing_citing)

    if missing_cited.size > 0:
        print("Missing cited IDs:", missing_cited)

    edges['citing'] = edges['citing'].map(id_to_index)
    edges['cited'] = edges['cited'].map(id_to_index)

    if edges['citing'].isna().any() or edges['cited'].isna().any():
        print("There are NaN values in the edges after mapping.")

    edges_list = list(zip(edges['citing'], edges['cited']))
    edges_list = [pair for pair in edges_list if all(not isinstance(x, float) or not (x != x) for x in pair)]

    return features_tensor, edges_list, class_tensor

def get_pokec_data(args):

    random.seed(42)

    papers = pd.read_parquet('data/pokec_features.parquet')

    print(f"Shape of papers DataFrame: {papers.shape}")

    num_words = 349
    papers.columns = [f'word_{i}' for i in range(1, num_words+1)]

    feature_tensor = torch.tensor(papers.values, dtype=torch.float32)

    edges = pd.read_parquet('data/pokec_edges.parquet')
    edges.columns = ['cited', 'citing']

    edges_list = list(zip(edges['citing'], edges['cited']))


    with open('data/data.pkl', 'rb') as file:
        list_of_neighbors = pickle.load(file)

    return feature_tensor, edges_list, list_of_neighbors
# Code for the ICLR 2025 submission: "GRAIN: Exact Graph Reconstruction from Gradients"
## Prerequisites

### Downloading the data

We have preprocessed the data we use in the non-chemical settings in a suitable format, which can be downloaded [here](https://mega.nz/folder/GhRzETiD#TsS5_d2wbBMbrW4grp179Q). These files should then be stored in the `\data` folder.

### Environment setup

- Install Anaconda. 
- Create the conda environment:<br>

> conda env create -f environment.yml -n grain

- Enable the created environment:<br>

> conda activate grain


### Running GRAIN on a finetuned model

In case you want to replicate our experiments on GRAIN against a finetuned GNN, please follow these instructions:

- Create the necessary necessary folders

> mkdir -p models

- Download the [pretrained model](https://mega.nz/file/CNpBFAoQ#BQai6AnWrgxbobIMjLhg5LqjEl5D3xr8MFEao6oX2uU) and the [original model](https://mega.nz/file/qN5zmC7Q#IxwMVI1QeCjcFoett0fz-VYT6JGUoaePBPDBxBUggoEand) and store them in the `\models` folder.


## Adapting the configuration

We provide a configuration file `config.yaml` that should be used for adapting the model architecture and some of the attack hyperparameters. Before each attack is run, please configure the file to fit for the experiment you are trying to replicate. Relevant parameters include:

- **MODEL**: The model architecture. We currently support only `gcn` and `gat`.

- **HIDDEN_SIZE**, **NODE_EMBEDDING_DIM**, **READOUT_HIDDEN_DIM**: the embedding dimensions for each GCN component, referred to as $d$ in the paper. We set them as equal for the sake of simplicity, but they can be changed separately. Default is 300.

- **N_LAYERS_BACKBONE**: The number of backbone layers of the GNN. Default is 2.

- **N_LAYERS_READOUT**: The number of readout layers of the GNN. Default is 2.

- **GRAPH_CLASSIFICATION**: A boolean tag describing whether the setting is graph or node classification (default is True).

- **ACT**: The activation function used in the GCN. Currently supports `relu` or `gelu`.

- **TOL_L**: The singular value tolerance that is used to determine the rank of the gradient at the given layer. We only use the defaults, but they can be updated if needed.

## Main GRAIN Experiments (Tables 2, 3, 8)

### Parameters

- **DATASET**: The dataset for the attack to be run on. Should be one of `zpn/tox21_srp53`, `zpn/clintox`, `zpn/bbbp`, `citeseer`, or `pokec`.

- **CONFIG_PATH**: The path to the configuration file. All relevant parameters can be adapted here.


### Optional parameters (for non-chemical experiments)

- **MIN_NODES**: The minimum amount of nodes that the sampled subraph should contain.

- **MAX_NODES**: The maximum amount of nodes that the sampled subgraph should contain.

- **N_INPUTS**: How many graphs should be sampled.

### Commands
For running the command on the chemical datasets, please use:

> ./scripts/attack.sh DATASET  --config_path CONFIG_PATH

The non-chemical datasets require subgraph sampling, that is regulated by the **MIN_NODES** and **MAX_NODES** parameters.
To replicate our results, the following splits should be used:
- 20 graphs between 1 and 10 nodes
- 40 graphs between 10 and 20 nodes
- 30 graphs between 20 and 30 nodes
- 10 graphs between 30 and 40 nodes

> ./scripts/attack.sh DATASET --config_path CONFIG_PATH --min_nodes MIN_NODES --max_nodes MAX_NODES --n_inputs N_INPUTS

The model architecture (GCN vs GAT) can be adapted in the configuration file.

## Citeseer ablations (Table 4)

### Parameters

- **MIN_NODES**: The minimum amount of nodes that the sampled subraph should contain.

- **MAX_NODES**: The maximum amount of nodes that the sampled subgraph should contain.

- **N_INPUTS**: How many graphs should be sampled.

### Commands

The citeseer dataset requires subgraph sampling, that is regulated by the **MIN_NODES** and **MAX_NODES** parameters.
To replicate our results, the following splits should be used:
- 20 graphs between 1 and 10 nodes
- 40 graphs between 10 and 20 nodes
- 30 graphs between 20 and 30 nodes
- 10 graphs between 30 and 40 nodes

For the experiment without using the degree assumption:

> ./scripts/attack.sh citeseer --use_degree --config_path CONFIG_PATH --min_nodes MIN_NODES --max_nodes MAX_NODES --n_inputs N_INPUTS

For the experiment without using the uniqueness heuristic:

> ./scripts/attack.sh citeseer --naive_build --config_path CONFIG_PATH --min_nodes MIN_NODES --max_nodes MAX_NODES --n_inputs N_INPUTS

The model architecture should be set to `gat` in the configuration file.

## Setting ablations (Table 5)

### Commands

All experiments are performed on the `zpn/tox21_srp53` dataset, on the default GCN architecture.

- For the GELU ablation, please change the `act` parameter inside the config to `gelu`.

- For the node classification task, please change the `graph_classification` parameter inside the config to `false`.

To replicate the results, please run:

> ./scripts/attack.sh zpn/tox21_srp53 --config_path CONFIG_PATH

## Model parameters experiments (Tables 10, 11)

### Parameters

- **CONFIG_PATH**: The path to the configuration file. All relevant parameters, namely the width and depth, can be adapted here. Make sure that the model architecture is a GCN.

### Commands
A single command is necessary to replicate the results:

> ./scripts/attack.sh zpn/tox21_srp53 --config_path CONFIG_PATH

## Large Graph Experiments (Table 12)

### Parameters

- **CONFIG_PATH**: The path to the configuration file. All relevant parameters can be adapted here. Ensure the model architecture is set to `gat`.

- **MIN_NODES**: The minimum amount of nodes that the sampled subraph should contain.

- **MAX_NODES**: The maximum amount of nodes that the sampled subgraph should contain.

### Commands

The `pokec` dataset requires subgraph sampling, that is regulated by the **MIN_NODES** and **MAX_NODES** parameters. To replicate our results, use the 25-30, 30-40, 40-50, and 50-60 splits, as shown in the paper.

> ./scripts/attack.sh pokec --config_path CONFIG_PATH --min_nodes MIN_NODES --max_nodes MAX_NODES --n_inputs 20

## Baseline Experiments (Tables 2, 3, 8, 9, 10)

We provide a modified verson of DLG and TabLeak inside the `\baseline` folder. They both utilise the aforementioned configuration file to determine the gcn structure.

### Parameters

- **DATASET**: The dataset for the attack to be run on. Should be one of `zpn/tox21_srp53`, `zpn/clintox`, `zpn/bbbp`, `citeseer`, or `pokec`.

- **CONFIG_PATH**: The path to the configuration file.

- **OPTIMIZER**: The optimizer to be used for the attack. Should be either `lbfgs` or `sgd`.

### Optional parameters (for non-chemical experiments)

- **MIN_NODES**: The minimum amount of nodes that the sampled subraph should contain.

- **MAX_NODES**: The maximum amount of nodes that the sampled subgraph should contain.

- **N_INPUTS**: How many graphs should be sampled.

### Commands (Chemical datasets)

To run DLG:

> ./scripts/dlg.sh DATASET --config_path CONFIG_PATH --optimizer OPTIMIZER

To run DLG with a given adjacency matrix:

> ./scripts/dlg.sh DATASET --config_path CONFIG_PATH --optimizer OPTIMIZER --fix_A

To run TabLeak:

> ./scripts/tableak.sh --config_path CONFIG_PATH DATASET --optimizer OPTIMIZER

To run TabLeak with a given adjacency matrix:

> ./scripts/tableak.sh DATASET --config_path CONFIG_PATH --optimizer OPTIMIZER --fix_A

Further changes (i.e. to the architecture) can be achieved through the configuration.

### Commands (Non-chemical datasets)

Similarly to the GRAIN experiments, the non-chemical datasets require subgraph sampling, that is regulated by the **MIN_NODES** and **MAX_NODES** parameters.
To replicate our results, the following splits should be used:
- 20 graphs between 1 and 10 nodes
- 40 graphs between 10 and 20 nodes
- 30 graphs between 20 and 30 nodes
- 10 graphs between 30 and 40 nodes

To run DLG:

> ./scripts/dlg.sh DATASET --config_path CONFIG_PATH --optimizer OPTIMIZER --min_nodes MIN_NODES --max_nodes MAX_NODES --n_inputs N_INPUTS

To run DLG with a given adjacency matrix:

> ./scripts/dlg.sh DATASET --config_path CONFIG_PATH --optimizer OPTIMIZER --min_nodes MIN_NODES --max_nodes MAX_NODES --n_inputs N_INPUTS --fix_A

To run TabLeak:

> ./scripts/tableak.sh --config_path CONFIG_PATH DATASET --optimizer OPTIMIZER --min_nodes MIN_NODES --max_nodes MAX_NODES --n_inputs N_INPUTS

To run TabLeak with a given adjacency matrix:

> ./scripts/tableak.sh DATASET --config_path CONFIG_PATH --optimizer OPTIMIZER --min_nodes MIN_NODES --max_nodes MAX_NODES --n_inputs N_INPUTS --fix_A

Further changes (i.e. to the architecture) can be achieved through the configuration.

## General notes

We recommend running the experiments using the [Neptune](https://neptune.ai/) framework, which can be enabled by specifying a workspace in the baseline scripts under the `--neptune` parameter, as this allows for all results to be viewed without printing overhead. For example:

> ./scripts/attack.sh zpn/clintox --neptune INSAIT/GRAIN

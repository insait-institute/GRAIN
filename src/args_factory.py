import argparse
import sys
import time
import os
import yaml

def get_args():
    parser = argparse.ArgumentParser()

    #Neptune configuration
    parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune', default=None)
    parser.add_argument('--label', type=str, default='name of the run', required=False)

    # Run configuration
    parser.add_argument('--rng_seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_inputs', default=100, type=int)
    parser.add_argument('--naive_build', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--do_ohe', action='store_false')


    # Dataset args
    parser.add_argument('--dataset', choices=['zpn/clintox', 'zpn/bbbp', 'zpn/tox21_srp53', 'citeseer', 'pokec'], required=True)
    
    parser.add_argument('--directed', action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_degree', default=10, type=int)
    parser.add_argument('--use_degree', action='store_false')
    parser.add_argument('--num_extra_features', default=0, type=int)
    parser.add_argument('--max_extra_value', default=3000, type=int)
    parser.add_argument('--min_nodes', default=0, type=int)
    parser.add_argument('--max_nodes', default=100, type=int)

    # Model configuration
    parser.add_argument('--config_path', type=str, default='./config.yaml')
    parser.add_argument('--eval_config_path', type=str, default='./config_eval.yaml') 
    parser.add_argument('--saved_model_path', type=str, default=None)   

    # FL environment arguments
    parser.add_argument('--federated_optimizer', type=str, default='FedSGD', choices=['FedSGD', 'FedAVG'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1)

    args=parser.parse_args(sys.argv[1:])
    run = None

    if args.neptune is not None:
        import neptune
        nep_par = { 'project':f"{args.neptune}", 'source_files':["*.py"] } 
        with open(args.config_path, 'r') as file:
            config = yaml.safe_load(file)
        label = f"grain_{args.dataset.split('/')[-1]}_{args.min_nodes}-{args.max_nodes}_l{config['model_args']['n_layers_backbone']}_d{config['model_args']['node_embedding_dim']}"
        
        if args.naive_build:
            label += '_naive'

        if args.num_extra_features > 0:
            label += f'_extra-{args.num_extra_features}'

        run = neptune.init_run(
            project=args.neptune, 
            api_token=os.environ['NEPTUNE_API_TOKEN'],
            name=label, 
            tags=["baseline"], 
            dependencies="infer", 
        )

        args_dict = vars(args)
        run[f"parameters"] = args_dict
        args.neptune = run
        print('waiting...')
        start_wait=time.time()
        args.neptune.wait()
        print('waited: ',time.time()-start_wait)
        args.neptune_id = args.neptune['sys/id'].fetch()
        print( '\n\n\nArgs:', *sys.argv[1:], '\n\n\n' ) 
        
    return args, run
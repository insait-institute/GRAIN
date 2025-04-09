import argparse
import sys
import time
import os

def get_args():
    parser = argparse.ArgumentParser()

    #Neptune configuration
    parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune', default=None)
    parser.add_argument('--label', type=str, default='name of the run', required=False)

    # Run configuration
    parser.add_argument('--rng_seed', type=int, default=42)
    parser.add_argument('--max_iter', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--do_ohe', action='store_false')
    parser.add_argument('--n_inputs', default=100, type=int)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lbfgs'], default='lbfgs')

    # Dataset args
    parser.add_argument('--dataset', choices=['zpn/clintox', 'zpn/bbbp', 'zpn/tox21_srp53', 'citeseer', 'pokec'], required=True)
    
    parser.add_argument('--directed', action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_degree', default=10, type=int)
    parser.add_argument('--use_degree', action='store_false')
    parser.add_argument('--num_extra_features', default=0, type=int)
    parser.add_argument('--max_extra_value', default=3000, type=int)
    parser.add_argument('--min_nodes', default=0, type=int)
    parser.add_argument('--max_nodes', default=100, type=int)
    
    # Hyperparams
    parser.add_argument('--A_thrs', type=float, default=0.5)
    
    # Model configuration
    parser.add_argument('--config_path', type=str, default='./config.yaml')
    parser.add_argument('--eval_config_path', type=str, default='./config_eval.yaml')
    
    #TabLeak config
    parser.add_argument('--n_ens', type=int, default=10)
        
    # Experiment configuration
    parser.add_argument('--fix_A', action=argparse.BooleanOptionalAction)
    parser.add_argument('--fix_X', action=argparse.BooleanOptionalAction)
    parser.add_argument('--fix_y', action=argparse.BooleanOptionalAction)

    args=parser.parse_args(sys.argv[1:])
    run = None

    if args.neptune is not None:
        import neptune
        assert('label' in args)
        nep_par = { 'project':f"{args.neptune}", 'source_files':["*.py"] } 

        run = neptune.init_run(
            project=args.neptune, 
            api_token=os.environ['NEPTUNE_API_KEY'],
            name=args.label, 
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
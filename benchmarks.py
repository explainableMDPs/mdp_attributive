import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

import pickle
from pathlib import Path
import argparse
import multiprocessing
import os
import random

seed = 42
random.seed(seed)


from Result import Result
from PrismParser import PrismParser

import pyrootutils
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(
path=path, # path to the root directory
project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
dotenv=True, # load environment variables from .env if exists in root directory
pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
cwd=True, # change current working directory to the root directory (helps with filepaths)
)

def get_parser(name):
    if name == 'greps':
        print("######### GREPS ##########")
        from LogParser import GrepsParser
        parser = GrepsParser('data/data.csv', 'data/activities_greps.xml')
    elif name == 'bpic12':
        print("######### BPIC'12 ##########")
        from LogParser import BPIC12Parser
        parser = BPIC12Parser('data/BPI_Challenge_2012.xes', 'data/activities_2012.xml')
    elif name == 'bpic17-before':
        print("######### BPIC'17-Before ##########")
        from LogParser import BPIC17BeforeParser
        parser = BPIC17BeforeParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
    elif name == 'bpic17-after':
        print("######### BPIC'17-After ##########")
        from LogParser import BPIC17AfterParser
        parser = BPIC17AfterParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
    elif name == 'bpic17-both':
        print("######### BPIC'17-Both ##########")
        from LogParser import BPIC17BothParser
        parser = BPIC17BothParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
    elif 'spotify' in name:
        print("######### Spotify ##########")
        from LogParser import SpotifyParser
        parser = SpotifyParser('data/spotify/', 'data/activities_spotify.xml', int(name.split('spotify')[1]))
    else:
        return None
    
    return parser
        
def generate_models(experiments, cores = 1, model_iterations = 1):
    parser_list = []
    for name in experiments:
        parser = get_parser(name)
        if not parser:
            continue
        
        if type(parser) != list:
            parser_list.extend([(parser, i, name) for i in range(model_iterations)])
        else:
            parser_list.extend([(p, i, name) for p in parser for i in range(model_iterations)])
                    
    #[build_and_write_model(p) for p in parser]
    with multiprocessing.Pool(processes=cores) as pool:
        pool.map(build_and_write_model, parser_list)
            
def build_and_write_model(input):
    parser = input[0]
    i = input[1]
    name = input[2]
    model = parser.build_benchmark(prism_name = f'out/models/model_{name}_model-it_{i}.prism')
    with open(f'out/models/model_{name}_model-it_{i}.pickle', 'wb+') as handle:
        pickle.dump(model, handle)
                    
def generate_paths(experiments, model_iterations, iterations, path_length_start=1, path_length_end=10):
    for name in experiments:
        for i in range(model_iterations):
            with open(f'out/models/model_{name}_model-it_{i}.pickle', 'rb') as handle:
                model = pickle.load(handle)
            parser = get_parser(name)
            for path_length in range(path_length_start, path_length_end+1):
                random_paths = list(nx.generate_random_paths(model, iterations, path_length=path_length, source='q0: start'))
                write_paths(f'out/paths/model_{name}_model-it_{i}_random_paths.txt', random_paths, append=(path_length!=path_length_start))
                print("random paths", random_paths)
                
                actual_paths = generate_actual_path(parser, model, iterations, path_length)
                write_paths(f'out/paths/model_{name}_model-it_{i}_actual_paths.txt', actual_paths, append=(path_length!=path_length_start))


def generate_actual_path(parser, model, iterations, path_length):
    
    traces = parser.get_data_trace(path_length, iterations)
    # transform trace into path in model with right state names
    constructed_paths = []
    for t in traces:
        s = 'q0: start'
        current_path = [s]
        for e in t[1:path_length+1]:
            possible_transitions = [i for i in model[s] if e[1] in i]
            assert len(possible_transitions) == 1, f'Error, more than one possible transition {possible_transitions} for {e}'
            current_path.append(possible_transitions[0])
            s = possible_transitions[0]
        constructed_paths.append(current_path)
    return constructed_paths

def write_paths(file_path, model_paths, append=False):
    with open(f'{file_path}', 'a+' if append else 'w+') as f:
        for path in model_paths:
            f.write(str(path)+'\n')

def run_experiment(param):
    if not args:
        diversity_runs = 4
    else:
        diversity_runs = args.diversity_runs
    
    path = param[0]
    p = param[1]
    if p == 0:
        p = 0.0001
    timeout = param[2]
    
    model_path = str(path).split('_it_')[0].replace('user_strategies', 'models')
    with open(f'{model_path}.pickle', 'rb') as handle:
        model = pickle.load(handle)
    with open(path, 'rb') as handle:
        user_strategy = pickle.load(handle)
    
    print(f'Call {model_path} with reachability probability {p} on strategy {path}')
    #r_qp = quadratic_program(model, p, user_strategy, timeout=timeout, debug=False)
    r_qp =  solver.QuadraticProblem(model, p, user_strategy, timeout=timeout, debug=False).solve()
    o, strat = minimum_reachability(model)
    
    # diversity run
    df_results_div = r_qp.df()
    df_results_div['id'] = 0
    df_results_div['path'] = path
    df_results_div['unknown_fraction'] = 1 if r_qp.status == GRB.OPTIMAL else 0
    # df_results_div['value'] = abs(r_qp.value - r_qp_new.value)
    
    if r_qp.status != GRB.OPTIMAL:
        return df_results_div
    
    results_div = [r_qp]
    for i in range(diversity_runs):
        print(f'strat {i}')
        #r_div = diversity_program_strategy(model, p, user_strategy, results_div, timeout=args.timeout, debug=False)
        r_div = solver.QuadraticProblem(model, p, user_strategy, timeout=timeout, debug=False).solve_diverse(results_div)
        if r_div.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            continue
        previously_chosen_actions = get_chosen_state_action(user_strategy, results_div)
        chosen_actions = get_chosen_state_action(user_strategy, [r_div])
        unknown_fraction = (len([a for a in chosen_actions if a not in previously_chosen_actions]) / len(chosen_actions)) if len(chosen_actions) != 0 else 0
        results_div.append(r_div)
        
        new_df = r_div.df()
        new_df['id'] = i+1
        new_df['path'] = path
        new_df['unknown_fraction'] = unknown_fraction
        # new_df['value'] = abs(r_div.value - r_div_new.value)
        df_results_div = pd.concat([df_results_div, new_df])

    return df_results_div

def get_chosen_state_action(user_strategy : dict, results : list):
    chosen_actions = set()
    for r in [r.strategy for r in results]:
        print(user_strategy.keys())
        print(r.keys())
        assert user_strategy.keys() == r.keys()
        for s in user_strategy:
            assert s in r
            assert user_strategy[s].keys() == r[s].keys()
            for a in user_strategy[s]:
                if round(user_strategy[s][a], 2) != round(r[s][a], 2):
                    chosen_actions.add((s,a))
    
    return chosen_actions

def reach_state(model, state):
    parser = PrismParser('/home/paul/Downloads/prism-4.8.1-linux64-x86/bin/prism', model)
    return parser.call_prism(""" Pmax=? [F """ + f'{state}' + "] """)
    # 'Pmax=? [F "positive"]'
    assert False

def manual_execution():
    assert args
    # manual tests
    with open('out/models/model_greps_model-it_0.pickle', 'rb') as handle: #open(f'out/models/model_{name}.pickle', 'rb') as handle:
        model = pickle.load(handle)
    # with open('out/user_strategies/model_spotify1000_model-it_6_it_3.pickle', 'rb') as handle:
        # user_strategy = pickle.load(handle)
        # user_strategy = pickle.load(handle)   
    # print("search_bounds", search_bounds(model, user_strategy))    
    
    reach_state('out/models/model_greps_model-it_0.prism', '"positive"')
    
    print("run exp")
    print(model.nodes())
    print("model length", len(model))
    results = [reach_state('out/models/model_greps_model-it_0.prism', f'loc={i}') for i in range(len(model.nodes())//10)]
    df_results = pd.DataFrame()
    stored_results = []
    for r in results:
        stored_results.append(r)
        df_results = pd.concat([df_results, r])
        df_results.to_csv("out/results.csv")
    # run_experiment((path, 0.35, args.timeout))
    print(stored_results)
    assert(False)
    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(
                    prog = 'benchmarks',
                    description = "File to trigger benchmarks for CE generation in MDP's")
    parser.add_argument('-t', '--timeout', help = "Timeout for Gurobi", type=int, default = 60*60) 
    parser.add_argument('-s', '--steps', help = "Number of steps for each model", type=int, default = 1)
    parser.add_argument('-i', '--iterations', help = "Iterations for each model", type=int, default = 1)
    parser.add_argument('-pls', '--path_length_start', help = "Start length for paths", type=int, default = 1)
    parser.add_argument('-ple', '--path_length_end', help = "End length for paths", type=int, default = 10)
    parser.add_argument('-c', '--cores', help = "Cores to use to parallelize experiments", type=int, default = 1)
    parser.add_argument('-e', '--experiments', help = "Name of experiments to run", nargs='+', type=str, default = ['greps', 'bpic12', 'bpic17-before', 'bpic17-after', 'bpic17-both', 'spotify'])
    parser.add_argument('-rm', '--rebuild_models', help = "Rebuild models, implies rebuilding strategies", action = 'store_true')
    parser.add_argument('-rs', '--rebuild_paths', help = "Rebuild paths for models", action = 'store_true')
    parser.add_argument('-mi', '--model_iterations', help = "Number of models to generate for each setting", type=int, default = 10)
    parser.add_argument('-as', '--all_spotify', help = "All spotify models in steps of 100 are generated", action = 'store_true')
    parser.add_argument('-d', '--diversity_runs', help = "Number of diverse counterfactuals", type=int, default = 0)
    parser.add_argument('-b', '--bounds', help="Bounds for gamma, evenly split by steps", nargs='+', type=float, default=[0, 1])
    args = parser.parse_args()
    
    if args.all_spotify:
        args.experiments.remove('spotify')
        args.experiments.extend([f'spotify{i*1000}' for i in range(1,11)])
    
    if args.rebuild_models:
        filename = "out/models/test.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        generate_models(args.experiments, args.cores, args.model_iterations)
    if args.rebuild_models or args.rebuild_paths:
        filename = "out/paths/test.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        generate_paths(args.experiments, args.model_iterations, args.iterations, path_length_start=args.path_length_start, path_length_end=args.path_length_end)
    
    # trigger single, manual execution
    manual_execution()
    
    # for name in args.experiments:
    benchmark_strategies = []
    for name in args.experiments:
        for i in range(args.iterations):
            for j in range(args.model_iterations):
                assert list(Path(f'out/user_strategies/').glob(f'*{name}_model-it_{j}*_it_{i}*.pickle'))
                benchmark_strategies.extend(list(Path(f'out/user_strategies/').glob(f'*{name}_model-it_{j}*_it_{i}*.pickle')))
    
    experiments = []
    for e in benchmark_strategies:
        path = str(e).split('_it_')[0].replace('user_strategies', 'models')
        name = str(e).split('model_')[1].split('_')[0]
        with open(f'{path}.pickle', 'rb') as handle: #out/models/model_{name}
            model = pickle.load(handle)
        with open(e, 'rb') as handle:
            user_strategy = pickle.load(handle)
        bounds = tuple(args.bounds)
        print(bounds)
        experiments.extend([(e, round(bounds[0] + (bounds[1] - bounds[0]) * 1/(args.steps) * s, 4), args.timeout) for s in range(args.steps+1)])
        #experiments.append((e, 1, args.timeout))
    # experiments = [(p, 1/(args.steps)*s, args.timeout) for p in benchmark_strategies for s in range(args.steps+1)]
    
    df_results = pd.DataFrame()
    stored_results = []
    with multiprocessing.Pool(processes=args.cores) as pool:
        result = pool.imap_unordered(run_experiment, experiments)
        for r in result:
            stored_results.append(r)
            df_results = pd.concat([df_results, r])
            df_results.to_csv("out/results_div.csv")
    # result = [run_experiment_diverse(e) for e in experiments]
    result = stored_results
    print("Done")
    
    
# TODO:
# iterate over all states - how to get different locations?
# build paths
# parallelize calls
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from gurobipy import GRB

import pickle
from pathlib import Path
import argparse
import multiprocessing
import os
import random
import ast
from functools import partial

seed = 42
random.seed(seed)


from Result import PrismResult, GurobiResult, GurobiResultLowerUpper
from PrismParser import PrismParser, StormParser
from solver import QuadraticEncoding, LinearEncoding, GeneralQuadraticEncoding

import pyrootutils
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(
path=path, # path to the root directory
project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
dotenv=True, # load environment variables from .env if exists in root directory
pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
cwd=True, # change current working directory to the root directory (helps with filepaths)
)

PRISM_PATH = ''
STORM_PATH = ''
TIMEOUT = 60 # in seconds
SOLVER = ''

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
    elif 'epidemic' in name:
        print("######### Epidemic ##########")
        from LogParser import EpidemicParser
        parser = EpidemicParser(int(name.split('epidemic')[1]))
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
                    
def generate_paths(experiments, model_iterations, iterations, path_length = (1,10,1)):
    for name in experiments:
        parser = get_parser(name)
        for i in range(model_iterations):
            with open(f'out/models/model_{name}_model-it_{i}.pickle', 'rb') as handle:
                model = pickle.load(handle)
            # add self-loop to be for generating arbitrarily long paths            
            terminal_states = [s for s in model.nodes() if 'positive' in s or 'negative' in s]
            assert len(terminal_states) == 2, f'Terminal states {terminal_states}'
            for s in terminal_states:
                model.add_edge(s, s) # add self-edge for path-generating purpose
            # max_length = min([nx.shortest_path_length(model, source='q0: start', target=s) for s in terminal_states])
            # print(max_length)
            for path_length_index in range(path_length[0], path_length[1], 1 if len(path_length) == 2 else path_length[2]):
                actual_paths = generate_actual_path(parser, model, iterations, path_length_index)
                write_paths(f'out/paths/model_{name}_model-it_{i}_actual_paths.txt', actual_paths, append=(path_length_index!=path_length[0]))
                
                random_paths = list(nx.generate_random_paths(model, iterations, path_length=path_length_index, source='q0: start'))
                # remove "q{i}: " state information
                random_paths = [[e.split(': ')[1] for e in t] for t in random_paths]
                write_paths(f'out/paths/model_{name}_model-it_{i}_random_paths.txt', random_paths, append=(path_length_index!=path_length[0]))
                

def contains_trace(model, s, trace):
    # assert trace[0] in s, f'State {s} does not contain trace-start {trace[0]}'
    if not trace:
        return True
    possible_transitions = [e[1] for e in model.out_edges(s) if model.edges[e]['action'] == trace[0][0] and trace[0][1] in e[1]]
    if not possible_transitions:
        return False
    return any([contains_trace(model, possible_transitions[i], trace[1:]) for i in range(len(possible_transitions))]) 

def generate_actual_path(parser, model, iterations, path_length):
    traces = parser.get_data_trace(path_length, iterations)
    # # keep generating traces until #iterations many are acutally contained in model
    # traces = []
    # while len(traces) < iterations:
    #     print("Constructing trace")
    #     generated_traces = parser.get_data_trace(path_length, 1)
    #     traces.extend([t for t in generated_traces if contains_trace(model, 'q0: start', t[1:])])
    # print("Constructed trace")
    constructed_paths = []
    for t in traces:
        current_path = ['start']
        for e in t[1:path_length+1]:
            current_path.append(e[1])
            # possible_transitions = [i for i in model[s] if e[1] in i]
            # assert len(possible_transitions) == 1, f'Error, more than one possible transition {possible_transitions} for {e}'
            # current_path.append(possible_transitions[0])
            # s = possible_transitions[0]
        constructed_paths.append(current_path)
    return constructed_paths

def write_paths(file_path, model_paths, append=False):
    with open(f'{file_path}', 'a+' if append else 'w+') as f:
        for path in model_paths:
            # remove here double terminal state again
            # path_no_doubles = [path[0]]
            # for i in range(1,len(path)):
            #     if path[i] != path[i-1]:
            #         path_no_doubles.append(path[i])
            f.write(str(path)+'\n')

def run_experiment(param):
    function = param[0]
    model = param[1]
    arg = param[2:]
    # name = param[3]
    # sense = param[4]
    
    print(f'Call {function} with model {model} and arg {arg}')

    return function(model, arg)

"""
State can be "positive" or loc={i}.
"""
def reach_state(model, state):
    parser = None
    if SOLVER == 'PRISM':
        parser = PrismParser(PRISM_PATH, model, TIMEOUT)
    if SOLVER == 'Storm':
        parser = StormParser(STORM_PATH, model, TIMEOUT)
    # 'Pmax=? [F "positive"]'
    return parser.call(""" Pmax=? [F """ + f'{state}' + "] """, "reach_state")

"""
State_id must be integer to not break encoding to PRISM.
"""
def avoid_positive_until_state(model, state_id):
    assert type(state_id) == int , f'Type of state_id {type(state_id)}'
    parser = None
    if SOLVER == 'PRISM':
        parser = PrismParser(PRISM_PATH, model, TIMEOUT)
    if SOLVER == 'Storm':
        parser = StormParser(STORM_PATH, model, TIMEOUT)
    # 'Pmax=? [F "positive" & (! "positive" U loc=state_reach)]'
    return parser.call(f'Pmax=? [(F "positive") & (!("positive") U loc={state_id})]', "avoid_state")

"""
Helper function to parse the label information at the end of the PRISM file.
"""
def get_label_dict(model):
    with open(model, 'rb') as f:
        labels = [line.decode("utf-8") for line in f]
        labels = [line.split('label ')[1].strip('\n') for line in labels if 'label ' in line] # also removes newline character from path
    label_dict = {}
    for l in labels:
        assigned_label = l.split(" = ")
        label_dict[assigned_label[0].replace('"', '')] = assigned_label[1].replace(';', '')
    return label_dict

"""
Formula encoding path recursively
"""
def follow_path(model, path, name=""):
    parser = None
    if SOLVER == 'PRISM':
        parser = PrismParser(PRISM_PATH, model, TIMEOUT)
    if SOLVER == 'Storm':
        parser = StormParser(STORM_PATH, model, TIMEOUT)
    # parse label from PRISM file
    label_dict = get_label_dict(model)
    if any([e not in label_dict for e in path]):
        print("WARNING: skipped path")
        return PrismResult(model, path, 0, -1, 0, 0, 0, name, TIMEOUT, "").df()
    
    def construct_path(inner_path):
        if len(inner_path) == 1:
            return f'X( {label_dict[inner_path[0]]} )'
        return f'(X( {label_dict[inner_path[0]]} & ({construct_path(inner_path[1:])})))'
    
    # print('constructed path', path, "-", construct_path(path[1:]))
    
    return parser.call(f'Pmax=? [(F "positive") & ({construct_path(path[1:])})]', name)

def importance_state(model, arg) -> GurobiResultLowerUpper:
    assert len(arg) == 2
    via_state = arg[0]
    name = arg[1]
    target_state = [s for s in model if 'positive' in s]
    assert len(target_state) == 1
    
    result_list = []
    
    qp = QuadraticEncoding(model, 'q0: start', via_state=via_state, target_state=target_state[0], debug=True, timeout=args.timeout)
    df_qp = qp.solve_lower_upper().df()
    df_qp.insert(0, 'name', [name])
    df_qp.insert(1, 'states', [str(len(model.nodes))])
    df_qp.insert(2, 'transitions', [str(len(model.edges))])
    df_qp.insert(3, 'encoding', ['QP'])
    result_list.append(df_qp)
    
    lp = LinearEncoding(model, 'q0: start', via_state=via_state, target_state=target_state[0], debug=True, timeout=args.timeout)
    df_lp = lp.solve_lower_upper().df()
    df_lp.insert(0, 'name', [name])
    df_lp.insert(1, 'states', [str(len(model.nodes))])
    df_lp.insert(2, 'transitions', [str(len(model.edges))])
    df_lp.insert(3, 'encoding', ['LP'])
    result_list.append(df_lp)
    
    # gqp = GeneralQuadraticEncoding(model, 'q0: start', via_state=via_state, target_state=target_state[0], debug=True, timeout=args.timeout)
    # df_gqp = gqp.solve_lower_upper().df()
    # df_gqp.insert(0, 'name', [name])
    # df_gqp.insert(1, 'states', [str(len(model.nodes))])
    # df_gqp.insert(2, 'transitions', [str(len(model.edges))])
    # df_gqp.insert(3, 'encoding', ['GQP'])
    # result_list.append(df_gqp)
    
    # return df_lp

    # if (df_lp['status'] == df_qp['status']).all() and False:
    #     # check for 0.0011 as 0.001 is precision, but is periodic (thus represented as 0.0010...01) -> compare against 0.0011
    #     vl = df_lp['lower_importance_value'].iloc[0]
    #     vq = df_qp['lower_importance_value'].iloc[0]
    #     assert (abs(df_lp['lower_importance_value'].iloc[0] - df_qp['lower_importance_value'].iloc[0]) <= 0.0011).all(), f'Error for "{name}" with via_state "{via_state}" (lower): {vl} != {vq}'
    #     vl = df_lp['upper_importance_value'].iloc[0]
    #     vq = df_qp['upper_importance_value'].iloc[0]
    #     assert (abs(df_lp['upper_importance_value'].iloc[0] - df_qp['upper_importance_value'].iloc[0]) <= 0.0011).all(), f'Error for "{name}" with via_state "{via_state}" (upper): {vl} != {vq}'
    df_merged = pd.concat(result_list, ignore_index=True, sort=False)

    return df_merged

def importance_state_sense(model, arg) -> GurobiResult:
    assert len(arg) == 3
    via_state = arg[0]
    name = arg[1]
    sense = arg[2]
    target_state = [s for s in model if 'positive' in s]
    assert len(target_state) == 1
    qp = QuadraticEncoding(model, 'q0: start', via_state=via_state, target_state=target_state[0], debug=True, timeout=args.timeout)
    df_qp = qp.solve(sense=sense).df()
    df_qp.insert(0, 'name', [name])
    df_qp.insert(1, 'states', [str(len(model.nodes))])
    df_qp.insert(2, 'transitions', [str(len(model.edges))])
    df_qp.insert(3, 'encoding', ['QP'])
    df_qp.insert(4, 'sense', [sense])
    
    lp = LinearEncoding(model, 'q0: start', via_state=via_state, target_state=target_state[0], debug=True, timeout=args.timeout)
    df_lp = lp.solve(sense=sense).df()
    df_lp.insert(0, 'name', [name])
    df_lp.insert(1, 'states', [str(len(model.nodes))])
    df_lp.insert(2, 'transitions', [str(len(model.edges))])
    df_lp.insert(3, 'encoding', ['LP'])
    df_lp.insert(4, 'sense', [sense])
    
    gqp = GeneralQuadraticEncoding(model, 'q0: start', via_state=via_state, target_state=target_state[0], debug=True, timeout=args.timeout)
    df_gqp = gqp.solve(sense=sense).df()
    df_gqp.insert(0, 'name', [name])
    df_gqp.insert(1, 'states', [str(len(model.nodes))])
    df_gqp.insert(2, 'transitions', [str(len(model.edges))])
    df_gqp.insert(3, 'encoding', ['GQP'])
    df_gqp.insert(4, 'sense', [sense])
    
    df_merged = pd.concat([df_qp, df_lp, df_gqp], ignore_index=True, sort=False)
    
    return df_merged
    
def manual_execution():
    assert args
    # manual tests
    with open('out/models/model_bpic17-before_model-it_0.pickle', 'rb') as handle: #open(f'out/models/model_{name}.pickle', 'rb') as handle:
        model = pickle.load(handle)
    target_state = [s for s in model if 'positive' in s]
    assert len(target_state) == 1
    qp = QuadraticEncoding(model, 'q0: start', 'q0: start', target_state[0], debug=True)
    qp.solve()
    
    with open('out/paths/model_greps_model-it_0_random_paths.txt', 'rb') as f:
        paths = [ast.literal_eval(line.decode("utf-8")) for line in f] # also removes newline character from path
    
    # reach_state('out/models/model_greps_model-it_0.prism', '"positive"')
    
    print("run exp")
    # results = [reach_state('out/models/model_greps_model-it_0.prism', f'loc={i}') for i in range(len(model.nodes()))]
    # results = [avoid_positive_until_state('out/models/model_greps_model-it_0.prism', i) for i in range(len(model.nodes()))]
    results = [follow_path('out/models/model_greps_model-it_0.prism', p, 'random_path') for p in paths]
    
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
    parser.add_argument('-t', '--timeout', help = "Timeout for PRISM (in sec.)", type=int, default = 60)
    parser.add_argument('-sa', '--samples', help = "Number of states to sample", type=int, default = 10)
    parser.add_argument('-i', '--iterations', help = "Iterations for each model", type=int, default = 1)
    parser.add_argument('-pl', '--path_length', help = "Path length range", type=int, nargs='+', default = [1,10,1])
    parser.add_argument('-c', '--cores', help = "Cores to use to parallelize experiments", type=int, default = 1)
    parser.add_argument('-e', '--experiments', help = "Name of experiments to run", nargs='+', type=str, 
                        default = ['greps', 'bpic12', 'bpic17-before', 'bpic17-after', 'bpic17-both', 'spotify', 'epidemic4'])
    parser.add_argument('-rm', '--rebuild_models', help = "Rebuild models, implies rebuilding strategies", action = 'store_true')
    parser.add_argument('-rs', '--rebuild_paths', help = "Rebuild paths for models", action = 'store_true')
    parser.add_argument('-mi', '--model_iterations', help = "Number of models to generate for each setting", type=int, default = 10)
    parser.add_argument('-as', '--all_spotify', help = "All spotify models in steps of 100 are generated", action = 'store_true')
    parser.add_argument('-pp', '--prism_path', help="Path to local PRISM executable", type=str, default='/home/ubuntu/prism-4.8.1-linux64-x86/bin/prism')
    parser.add_argument('-sp', '--storm_path', help="Path to local Storm executable", type=str, default='/home/ubuntu/storm-stable/build/bin/storm')
    parser.add_argument('-s', '--solver', help="Which solver to choose from [PRISM, Storm]", type=str, default='PRISM')
    args = parser.parse_args()
    
    assert len(args.path_length) in [2,3], f'Wrong length for Range function: 2 or 3 elements.'
    
    # set global PRISM path
    PRISM_PATH = args.prism_path
    STORM_PATH = args.storm_path
    TIMEOUT = args.timeout
    SOLVER = args.solver
    
    if args.all_spotify:
        args.experiments.remove('spotify')
        args.experiments.extend([f'spotify{i*1000}' for i in range(1,11)])
    
    if args.rebuild_models:
        filename = "out/models/test.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        generate_models(args.experiments, args.cores, args.model_iterations)
        
    # if args.rebuild_models or args.rebuild_paths:
    #     filename = "out/paths/test.txt"
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     generate_paths(args.experiments, args.model_iterations, args.iterations, tuple(args.path_length))
    
    # trigger single, manual execution
    # manual_execution()
    
    benchmark_models = []
    for name in args.experiments:
        for i in range(args.model_iterations):
            assert list(Path(f'out/models/').glob(f'*{name}_model-it_{i}.pickle'))
            benchmark_models.extend(list(Path(f'out/models/').glob(f'*{name}_model-it_{i}.pickle')))
    print('Benchmarks: ', [str(b) for b in benchmark_models])
    
    experiments = []
    for e in benchmark_models:
        print(e)
        with open(e, 'rb') as handle: # need pickle files for nodes
            model = pickle.load(handle)
        if "epidemic" in str(e):
            max_pop = int(str(e).replace('out/models/model_epidemic', '').replace('_model-it_0.pickle', ''))
            vacc_states = list([s for s in model.nodes() if s[0] == max_pop and s[1] == 1 and nx.has_path(model, 'q0: start', 'positive')])
            print("For model ", model, "there are", len(vacc_states), "states")
            experiments.extend([(importance_state_sense, model, s, e, GRB.MINIMIZE) for s in random.sample(vacc_states, k = min(args.samples, len(vacc_states)))])
            experiments.extend([(importance_state_sense, model, s, e, GRB.MAXIMIZE) for s in random.sample(vacc_states, k = min(args.samples, len(vacc_states)))])
        else:
            experiments.extend([(importance_state_sense, model, s, e, GRB.MINIMIZE) for s in random.sample(list(model.nodes()), k = min(args.samples, len(model.nodes)))])
            experiments.extend([(importance_state_sense, model, s, e, GRB.MAXIMIZE) for s in random.sample(list(model.nodes()), k = min(args.samples, len(model.nodes)))])
        continue
        # experiments.extend([(reach_state, e, f'loc={s}') for s in random.sample(list(range(len(model.nodes()))), k = args.samples)])
        # experiments.extend([(avoid_positive_until_state, e, s) for s in random.sample(list(range(len(model.nodes()))), k = args.samples)])
        # experiments.extend([(reach_state, e, f'loc={s}') for s in range(len(model.nodes()))])
        # experiments.extend([(avoid_positive_until_state, e, s) for s in range(len(model.nodes()))])
        
        path = str(e).split('_it_')[0].replace('user_strategies', 'models')
        name = str(e).split('model_')[1].split('_')[0]
        
        random_path_file = str(e).replace(".prism","").replace("models", "paths")+'_random_paths.txt'
        assert Path(random_path_file)
        with open(random_path_file, 'rb') as f:
            paths = [ast.literal_eval(line.decode("utf-8")) for line in f]
        experiments.extend([(partial(follow_path, name='random_path'), e, p) for p in paths])
        
        actual_path_file = str(e).replace(".prism","").replace("models", "paths")+'_actual_paths.txt'
        assert Path(actual_path_file)
        with open(actual_path_file, 'rb') as f:
            paths = [ast.literal_eval(line.decode("utf-8")) for line in f]
        experiments.extend([(partial(follow_path, name='actual_path'), e, p) for p in paths])

    print(f'########## Start computation for {len(experiments)} experiments')
    
    df_results = pd.DataFrame()
    stored_results = []
    with multiprocessing.Pool(processes=args.cores) as pool:
        result = pool.imap_unordered(run_experiment, experiments)
        for r in result:
            stored_results.append(r)
            df_results = pd.concat([df_results, r])
            df_results.to_csv("out/results.csv")
    # result = [run_experiment_diverse(e) for e in experiments]
    result = stored_results
    print("Done")
    
    
# TODO: current path construction breaks for spotify - not sure that paths are in sub-set contained
# TODO: all (actual) paths != 0 probability
# TODO: test prism settings - e.g. maxiters etc.


# TODO check learning for MultiDiGraph problem - they return just DiGraph
# TODO clean benchmarks setting here
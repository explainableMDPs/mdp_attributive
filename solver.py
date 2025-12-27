import networkx as nx
import gurobipy as gp
from gurobipy import GRB

import random
random.seed(42)

from Result import GurobiResult, GurobiResultLowerUpper
from fixed_mdp import *

def unroll(model : nx.MultiDiGraph, via_state : str, start_state = None) -> nx.MultiDiGraph:
    """Function to unroll model around via_state and, if start_state is given, prune all states that can not be reached from start_state.

    Args:
        model (nx.MultiDiGraph): Input MDP
        via_state (str): State to roll around
        start_state (str, optional): Initial state in MDP. Defaults to None.

    Returns:
        nx.MultiDiGraph: Unrolled MDP
    """
    if start_state:
        if via_state == start_state:
            unrolled_model = nx.MultiDiGraph(model)
            unrolled_model = nx.relabel_nodes(unrolled_model, {s : (s, 't') for s in model.nodes})
            # IMPORTANT: Rename here, later every computation is fixed as starting from '(start_state, f)'
            unrolled_model = nx.relabel_nodes(unrolled_model, {(start_state, 't') : (start_state, 'f') })
            return unrolled_model
        
    unrolled_model = nx.MultiDiGraph()
    unrolled_model.add_nodes_from([(n, 'f') for n in model.nodes])
    unrolled_model.add_nodes_from([(n, 't') for n in model.nodes])
    for e in model.edges:
        # e is a 3-tuple that includes the key (u,v,key) - model.edges[e] is possible
        if e[1] == via_state:
            unrolled_model.add_edge((e[0], 'f'), (e[1], 't'), action = model.edges[e]['action'], prob_weight= model.edges[e]['prob_weight'])
            unrolled_model.add_edge((e[0], 't'), (e[1], 't'), action = model.edges[e]['action'], prob_weight= model.edges[e]['prob_weight'])
        else:
            unrolled_model.add_edge((e[0], 'f'), (e[1], 'f'), action = model.edges[e]['action'], prob_weight= model.edges[e]['prob_weight'])
            unrolled_model.add_edge((e[0], 't'), (e[1], 't'), action = model.edges[e]['action'], prob_weight= model.edges[e]['prob_weight'])
    
    if start_state:
        reachable_nodes = list(nx.dfs_postorder_nodes(unrolled_model,source=(start_state,'f')))
        unrolled_model.remove_nodes_from([s for s in unrolled_model.nodes if s not in reachable_nodes])
    return unrolled_model
    
def add_self_loops(model: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Adds deterministic self-loop to all terminal states

    Args:
        model (nx.MultiDiGraph): Input MDP

    Returns:
        nx.MultiDiGraph: Output MDP
    """
    for s in model.nodes:
        if len(model[s]) == 0:
            model.add_edge(s, s, action = 'self_loop', prob_weight=1)
    return model
    
def ensure_optimal_reachability():
    #TODO - Implement to run with experiments that reachability solving is correct
    pass
    
    
class FixedReachabilitiesReturns:
    def __init__(self, fixed_reachabilities : dict, reachability : float, runtime : float):
        self.fixed_reachabilities = fixed_reachabilities
        self.reachability = reachability
        self.runtime = runtime
        
        
def get_fixed_reachabilities(model : nx.MultiDiGraph, start_state : str, via_state : str, target_state : str, timeout = 10*60*60, threads = 1, debug = False, memory=4, precision = 1e-2) -> FixedReachabilitiesReturns:
    env = gp.Env()
    m = gp.Model("lp", env=env)
    m.setParam('TimeLimit', timeout)
    m.setParam('SoftMemLimit', memory)
    m.setParam('Threads', threads)
    
    model = unroll(add_self_loops(model), via_state, start_state)
    
    p_s = {s : m.addVar(ub=1.0, name=f'p_{str(s)}', lb = 0.0) for s in model.nodes}
    
    target_states_unrolled = []
    if (target_state, 'f') in model:
        target_states_unrolled.append((target_state, 'f'))
        m.addConstr(p_s[(target_state, 'f')] == 1)
    if (target_state, 't') in model:
        target_states_unrolled.append((target_state, 't'))
        m.addConstr(p_s[(target_state, 't')] == 1)
    
    states_reaching = [s for s in model.nodes() if any([nx.has_path(model, s, t) for t in target_states_unrolled])]
    for s in model.nodes:
        print(s)
        if s not in states_reaching:
            m.addConstr(p_s[s] == 0)
        else:
            enabled_actions = set([model.edges[e]['action'] for e in list(model.edges(s, keys=True))])
            for action in enabled_actions:
                print('-- action', action)
                m.addConstr(p_s[s] >= sum([float(model.edges[e]['prob_weight']) * p_s[e[1]] for e in list(model.edges(s, keys=True)) if model.edges[e]['action'] == action]))
    
    m.setObjective(sum([p_s[s] for s in states_reaching]))
    
    m.ModelSense = GRB.MINIMIZE
    # self.m.setObjective(self.goal_var, sense = sense)
        
    # TODO ensure that model is linear
    m.update()
    m.optimize()    
    assert m.status == GRB.OPTIMAL, f'Status is {m.status} instead op OPTIMAL'
    print("Reachability", p_s[(start_state, 'f')].X)
    
    if debug:
        for v in m.getVars():
            print(f"{v.VarName} {v.X:g}")
        print(f"Obj: {m.ObjVal:g}")
        
    # compare optimal decisions
    fixed_reachabilities = {}
        
    for s in model.nodes():
        out_edges = list(model.edges(s, keys=True))
        # ERROR: Have to look on action basis, not per transition... - otherwise, env always sets only transition to 0
        if debug:
            print('Out edges from', s, ':', out_edges)
        enabled_actions = set([model.edges[e]['action'] for e in out_edges])
        if debug:
            print('Enabled actions', enabled_actions)
        maximizing_actions = [a for a in enabled_actions if sum([model.edges[e]['prob_weight'] * p_s[e[1]].X for e in out_edges if model.edges[e]['action'] == a]) >= p_s[s].X - precision]
        # maximizing_transitions = [e for e in out_edges if p_s[e[1]].X >= p_s[e[0]].X - precision]
        assert maximizing_actions
        if debug:
            print("maximizing actions", maximizing_actions)
        for a in enabled_actions:
            if a not in maximizing_actions:
                if debug:
                    print(f'Added {(s, a)} = 0 as {a} not in maximizing ({[model.edges[e]["prob_weight"] * p_s[e[1]].X for e in out_edges if model.edges[e]["action"] == a]} !>= {p_s[s].X} - {precision})')
                    fixed_reachabilities[(s, a)] = 0
        # TODO: should be redundant constraint 
        if len(maximizing_actions) == 1:
            if debug:
                print(f'Added {(s, maximizing_actions[0])} = 1 as {a} is only maximizing from {enabled_actions}')
            fixed_reachabilities[(s, maximizing_actions[0])] = 1
        # for e in out_edges:
        #     if e not in maximizing_transitions:
        #         if debug:
        #             print(f'Added {(s, model.edges[e]["action"])} = 0 as {e} not in maximizing ({p_s[e[1]].X} !>= {p_s[e[0]].X} - {precision})')
        #         fixed_reachabilities[(s, model.edges[e]['action'])] = 0
        # # not necessary, sum ensures already
        # if len(maximizing_transitions) == 1:
        #     if debug:
        #         print(f'Added {(s, model.edges[maximizing_transitions[0]]["action"])} = 1 as {s} is only maximizing')
        #     fixed_reachabilities[(s, model.edges[maximizing_transitions[0]]['action'])] = 1
    if debug:
        print("fixed", fixed_reachabilities)
    
    reachability_value = p_s[(start_state, 'f')].X
    print("Reachability is", reachability_value)
    runtime_value = m.Runtime
    m.dispose()
    return FixedReachabilitiesReturns(fixed_reachabilities, reachability_value, runtime_value)


class QuadraticProblem:
    def __init__(self, model : nx.MultiDiGraph, start_state : str, via_state : str, target_state : str, timeout = 10*60*60, threads = 1, debug = False, memory=4, precision = 1e-4):  
        # compute reachabilities
        self.fixed_reachabilities_return = get_fixed_reachabilities(model=model, start_state=start_state, via_state=via_state, target_state=target_state, timeout=timeout, threads=threads, debug=debug, memory=memory)
        
        self.env = gp.Env()
        self.m = gp.Model("qp", env=self.env)
        self.m.setParam('TimeLimit', timeout)
        self.m.setParam('SoftMemLimit', memory)
        self.m.setParam('Threads', 6)
        
        self.model = unroll(add_self_loops(model), via_state, start_state)
        self.start_state = start_state
        self.via_state= via_state
        self.target_state = target_state
        self.timeout = timeout
        self.debug = debug

        assert start_state in model        
        assert target_state in model
        assert via_state in model 
        
        self.p_s_t = {s : self.m.addVar(ub=1.0, name=f'p_{str(s)}->t', lb = 0.0) for s in self.model.nodes}
        self.p_s_f = {s : self.m.addVar(ub=1.0, name=f'p_{str(s)}->f', lb = 0.0) for s in self.model.nodes}
        self.p_sa = {}

        # start_state = [s for s in self.model.nodes if 'q0: start' in s]
        # assert len(start_state) == 1, start_state
        # self.start_state = start_state[0]
        
        # target_state = [s for s in self.model.nodes if 'negative' in s]
        # assert len(target_state) == 1, target_state
        # self.target_state = target_state[0]
        if (self.target_state, 'f') not in self.model.nodes:
            print('###### No negative contained ######')
            self.reaching_states = [s for s in self.model.nodes if s[0] != self.target_state and nx.has_path(self.model, s, (self.target_state, 't'))]
            self.m.addConstr(self.p_s_f[(self.start_state, 'f')] == 0) # no (target, 'f') state contained, thus, initial state can be set to 0 -> if no target state, the propagation can get lost
            self.m.addConstr(self.p_s_f[(self.target_state, 't')] == 0)
            self.m.addConstr(self.p_s_t[(self.target_state, 't')] == 1)
        elif (self.target_state, 't') not in self.model.nodes:
            print('###### No positive contained ######')
            self.reaching_states = [s for s in self.model.nodes if s[0] != self.target_state and nx.has_path(self.model, s, (self.target_state, 'f'))]
            self.m.addConstr(self.p_s_t[(self.start_state, 'f')] == 0) # no (target, 't') state contained, thus, initial state can be set to 0 -> if no target state, the propagation can get lost
            self.m.addConstr(self.p_s_t[(self.target_state, 'f')] == 0)
            self.m.addConstr(self.p_s_f[(self.target_state, 'f')] == 1)
        else:
            print('###### Both contained ######')
            self.reaching_states = [s for s in self.model.nodes if s[0] != self.target_state and (
                                    ((not (self.target_state, 'f')) or (nx.has_path(self.model, s, (self.target_state, 'f'))))
                                    or nx.has_path(self.model, s, (self.target_state, 't')))]
            self.m.addConstr(self.p_s_f[(self.target_state, 'f')] == 1)
            self.m.addConstr(self.p_s_f[(self.target_state, 't')] == 0)
            self.m.addConstr(self.p_s_t[(self.target_state, 'f')] == 0)
            self.m.addConstr(self.p_s_t[(self.target_state, 't')] == 1)
                
        if debug:
            print("Reaching states", self.reaching_states)
        # default values

        
        self.encode_actions()
        self.encode_model()
        
        self.goal_var = self.m.addVar(ub=1.0, name='goal variable', lb = 0)
        self.m.addConstr(self.goal_var*(self.p_s_t[(self.start_state, 'f')] + self.p_s_f[(self.start_state, 'f')]) == self.p_s_t[(self.start_state, 'f')])
        # self.m.addConstr(self.p_s_t[(self.start_state, 'f')] + self.p_s_f[(self.start_state, 'f')] >= 0.001)
        
    def encode_actions(self) -> dict:
        # encode actions - only for states that can reach the terminal state
        for s in self.model.nodes:
            # Encode pos and neg states as absorbing, i.e. without available actions. Thus, the can not be included in p_sa
            # print(self.model.edges[list(self.model.out_edges(s))[0]]['action'])
            enabled_actions = set([self.model.edges[e]['action'] for e in list(self.model.edges(s, keys=True))])
            print(f'enabled from {s} : {enabled_actions}')
            assert len(enabled_actions) >= 1, f'State{s} has no enabled action'
            self.p_sa[s] = {a : self.m.addVar(ub=1.0, name=str(s)+'_'+a, lb = 0, vtype=GRB.BINARY) for a in enabled_actions} # 
            self.m.addConstr(sum(list(self.p_sa[s].values())) == 1) # scheduler sums up to one
            for a in enabled_actions:
                self.m.addConstr(self.p_sa[s][a] <= 1)
        # parse fixed reachability values into model
        self.m.update() # update to parse variable names
        for e in self.fixed_reachabilities_return.fixed_reachabilities:
            self.m.addConstr(self.p_sa[e[0]][e[1]] == self.fixed_reachabilities_return.fixed_reachabilities[e])
            if self.debug:
                print(f'From pre-processing, added {self.p_sa[e[0]][e[1]].VarName} = {self.fixed_reachabilities_return.fixed_reachabilities[e]} to model')
             
    def encode_model(self):
        # encode model
        for s in self.p_sa:
            enabled_actions = set([self.model[e[0]][e[1]][k]['action'] for e in list(self.model.edges(s)) for k in self.model[s][e[1]]])
            assert len(enabled_actions) >= 1, f'State{s} has no enabled action'
            if s in self.reaching_states:
                self.m.addConstr(self.p_s_t[s] == sum([self.p_sa[s][self.model.edges[e]['action']] * float(self.model.edges[e]['prob_weight']) * self.p_s_t[e[1]] for e in list(self.model.edges(s, keys=True))]))
                self.m.addConstr(self.p_s_f[s] == sum([self.p_sa[s][self.model.edges[e]['action']] * float(self.model.edges[e]['prob_weight']) * self.p_s_f[e[1]] for e in list(self.model.edges(s, keys=True))]))
            else:
                if self.target_state not in s[0]:
                    # Not reachable states are still in strategy - exclude other target states
                    if self.debug:
                        print(f'Set {s} to 0')
                    self.m.addConstr(self.p_s_t[s] == 0)
                    self.m.addConstr(self.p_s_f[s] == 0)
                
    def get_max_solution(self, op):
        if self.debug:
            print(f"Found {self.m.SolCount} solutions:")
        solutions = []
        reachability_scores = []
        relevance_scores = []

        for s in range(self.m.SolCount):
            # Set which solution we will query from now on
            self.m.params.SolutionNumber = s
            reachability_scores.append(self.p_s_t[(self.start_state, 'f')].Xn + self.p_s_f[(self.start_state, 'f')].Xn)
            relevance_scores.append(self.goal_var.Xn)
            
            if self.debug:
                print('Solution', s, ':', end='')
            solution = []
            for o in range(self.m.NumObj):
                # Set which objective we will query
                self.m.params.ObjNumber = o
                # Query the o-th objective value
                if self.debug:
                    print(' ', self.m.ObjNVal, end='')
                solution.append(abs(self.m.ObjNVal)) # append abs to consider positive values
            # Print first three variables in the solution
            solutions.append(tuple(solution))
            if self.debug:
                print('')
            if self.debug:
                print("Reachability", self.p_s_t[(self.start_state, 'f')].Xn + self.p_s_f[(self.start_state, 'f')].Xn)
                print("Relevance", self.goal_var.Xn)
        reachability_scores = [e for e in reachability_scores]
        relevance_scores = [e for e in relevance_scores]

        arg_index = [i for i in range(len(reachability_scores)) if reachability_scores[i] >= op(reachability_scores)]
        op_relevance_score = [i for i in arg_index if relevance_scores[i] >= op([relevance_scores[j] for j in arg_index])]
        
        return (reachability_scores[op_relevance_score[0]], relevance_scores[op_relevance_score[0]])
        
    def get_solution(self, op):
        if self.m.status == GRB.INFEASIBLE:
            return GurobiResult(time=self.m.Runtime, reachability_value=-0.2, importance_value=-0.2, start_state=self.start_state, via_state=self.via_state, target_state=self.target_state, timeout=self.timeout, status=self.m.status)
        
        # compute result as in diverse target function include determinant
        if self.m.status == GRB.TIME_LIMIT:
            if self.m.SolCount == 0:
                return GurobiResult(time=self.m.Runtime, reachability_value=0, importance_value=0, start_state=self.start_state, via_state=self.via_state, target_state=self.target_state, timeout=self.timeout, status=self.m.status)
            else:
                max_result = self.get_max_solution(op)
                return GurobiResult(time=self.m.Runtime, reachability_value=max_result[0], importance_value=max_result[1], start_state=self.start_state, via_state=self.via_state, target_state=self.target_state, timeout=self.timeout, status=self.m.status)
        max_result = self.get_max_solution(op) 
        return GurobiResult(time=self.m.Runtime, reachability_value=max_result[0], importance_value=max_result[1], start_state=self.start_state, via_state=self.via_state, target_state=self.target_state, timeout=self.timeout, status=self.m.status)


    def solve_helper(self, sense=GRB.MAXIMIZE):
        # if sense == GRB.MAXIMIZE:
        # Higher priority is solved first ... while imposing constraints that ensure that
        # the quality of higher-priority objectives isn’t degraded by more than the specified tolerance
        # (https://docs.gurobi.com/projects/optimizer/en/current/features/multiobjective.html#secmultipleobjectives)
        # maximize reachability, then optimize for importance
        self.m.setObjectiveN(self.p_s_t[(self.start_state, 'f')] + self.p_s_f[(self.start_state, 'f')], index = 0, priority = 1) 
        # else:
            # self.m.setObjectiveN(-(self.p_s_t[(self.start_state, 'f')] + self.p_s_f[(self.start_state, 'f')]), index = 0, priority = 1)
            
        # Idea: Optimize for importance among all optimal strategies, then set remaining variables to 0
        if sense == GRB.MAXIMIZE:
            self.m.setObjectiveN(self.goal_var, index = 1, priority=0)
        else:
            self.m.setObjectiveN(-self.goal_var, index = 1, priority=0)
            
        self.m.ModelSense = GRB.MAXIMIZE
        # self.m.setObjective(self.goal_var, sense = sense)
        
        self.m.update()
        
        self.m.optimize()
        # print(self.m.display())
        
        return_result = self.get_solution(max if sense == GRB.MAXIMIZE else min)
        if self.m.status == GRB.INFEASIBLE or self.m.status == GRB.TIME_LIMIT:
            if self.m.Status == GRB.INFEASIBLE:
                self.m.computeIIS()
                print('\nThe following constraints and variables are in the IIS:')
                for c in self.m.getConstrs():
                    if c.IISConstr: print(f'\t{c.constrname}: {self.m.getRow(c)} {c.Sense} {c.RHS}')

                for v in self.m.getVars():
                    if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
                    if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
    
            assert False
            self.m.dispose()
            return return_result
        
        assert self.m.status == GRB.OPTIMAL, f'Status is {self.m.status}'
        print("Reachability", self.p_s_t[(self.start_state, 'f')].X + self.p_s_f[(self.start_state, 'f')].X)
        print("Relevance", self.goal_var.X)
        if self.debug:
            for v in self.m.getVars():
                print(f"{v.VarName} {v.X:g}")
            print(f"Obj: {self.m.ObjVal:g}")

        assert self.p_s_t[(self.start_state, 'f')].X + self.p_s_f[(self.start_state, 'f')].X != 0, f'Denominator is valued at 0'
        assert self.p_s_t[(self.start_state, 'f')].X + self.p_s_f[(self.start_state, 'f')].X <= 1, f'Reachability is larger than 1 : {self.p_s_t[(self.start_state, "f")].X + self.p_s_f[(self.start_state, "f")].X}'
        assert abs(self.p_s_t[(self.start_state, 'f')].X + self.p_s_f[(self.start_state, 'f')].X - self.fixed_reachabilities_return.reachability) <= 0.01, f'Reachability differs by more than 0.01 : {self.p_s_t[(self.start_state, "f")].X + self.p_s_f[(self.start_state, "f")].X} != {self.fixed_reachabilities_return.reachability}'


        return return_result
    
    def solve_lower_upper(self):
        return_result_lower = self.solve_helper(sense=GRB.MINIMIZE)
        return_result_upper = self.solve_helper(sense=GRB.MAXIMIZE)        
        
        assert return_result_lower.start_state == return_result_upper.start_state
        assert return_result_lower.via_state == return_result_upper.via_state
        assert return_result_lower.target_state == return_result_upper.target_state
        assert return_result_lower.timeout == return_result_upper.timeout
        assert return_result_lower.status == return_result_upper.status
        assert abs(return_result_lower.reachability_value - return_result_upper.reachability_value) <= 1e-3, f'{abs(return_result_lower.reachability_value - return_result_upper.reachability_value)}'
        assert return_result_lower.importance_value <= return_result_upper.importance_value
        
        result_lower_upper = GurobiResultLowerUpper(return_result_lower.time + return_result_upper.time, return_result_lower.reachability_value, return_result_lower.importance_value, 
                                                    return_result_upper.reachability_value, return_result_upper.importance_value, 
                                                    return_result_lower.start_state, return_result_lower.via_state, return_result_lower.target_state, 
                                                    return_result_lower.timeout, return_result_lower.status)
        
        self.m.dispose()
        return result_lower_upper
    
    def solve(self, sense=GRB.MAXIMIZE):
        return_result = self.solve_helper(sense=sense)
        self.m.dispose()
        return return_result


def gridworld_experiment():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image
    import numpy as np
    
    mdp = gridworld_mdp()
    
    get_fixed_reachabilities(mdp, 's00', 's33', 's60', debug=True)
    # assert False
    
    pos = {s : (int(s[1]), int(s[2])) for s in mdp.nodes()}
    nx.draw(mdp, pos, with_labels=True, node_size=0)
    plt.savefig("out/mdp.png")
    plt.cla()
    
    # unrolled = unroll(add_self_loops(mdp), 's00', 's66')
    qp = QuadraticProblem(mdp, 's00', 's11', 's66', debug=True)
    r = qp.solve_lower_upper().df()
    print(r)
    # assert False
    
    import pandas as pd 
    df_results = pd.DataFrame()
    for s in mdp.nodes():
        qp = QuadraticProblem(mdp, 's00', s, 's60', debug=True)
        r = qp.solve_lower_upper().df()
        df_results = pd.concat([df_results, r])
    df_results.to_csv("out/results.csv")
    
    print(df_results[df_results['via_state']==f's{3}{3}'].iloc[0][['lower_importance_value', 'upper_importance_value']])

    # use colormap
    print(plt.cm.jet(0))
    print(plt.cm.jet(1))
    
    w, h = 7, 7
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(7):
        for j in range(7):
            print(i,j)
            h = df_results[df_results['via_state']==f's{i}{j}'].iloc[0][['lower_importance_value', 'upper_importance_value']]
            print(h)
            print(data[j,i])
            # data[i,j] = [1,1,1]
            # data[i,j] = [int(plt.cm.jet(h['lower_importance_value'])[0]*255), 0, int(plt.cm.jet(h['upper_importance_value'])[2]*255)]
            data[j,i] = [255, int(h['lower_importance_value']*255), int(h['upper_importance_value']*255)]
            print(data[i,j])
    # data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
    print(data)
    img = Image.fromarray(data)
    # img = img.resize((700,700),resample=Image.NEAREST)
    img.save('out/mdp_results.png')
    
    elements = set(zip(df_results['lower_importance_value'], df_results['upper_importance_value']))
    patches = []
    for e in elements:
        patches.append(mpatches.Patch(color=[1, int(e[0]), int(e[1])], label=e))
    plt.legend(handles=patches)
    plt.imread('out/mdp_results.png')
    # plt.imsave('out/mdp_results.png', img, cmap='gray')
    imgplot = plt.imshow(img, aspect='equal')
    plt.savefig('out/mdp_results.png', bbox_inches='tight', dpi=200)
    
def loop_example():
    mdp = loop_mdp()
    
    get_fixed_reachabilities(mdp, 'a', 'b', 'c', debug=True)
    
    qp = QuadraticProblem(mdp, 'a', 'b', 'c', debug=True)
    r = qp.solve_lower_upper().df()
    print(r)
    assert (r['lower_reachability_value'] == r['upper_reachability_value']).all()
    assert r['lower_reachability_value'].iloc[0] == 1, r['lower_reachability_value'].iloc[0]
    assert r['lower_importance_value'].iloc[0] == 0, r['lower_importance_value'].iloc[0]
    assert r['upper_importance_value'].iloc[0] == 1, r['lower_importance_value'].iloc[0]
    
    return r

def loop_example_half():
    mdp = loop_half_mdp()
    
    get_fixed_reachabilities(mdp, 'a', 'b', 'pos', debug=True)
    
    qp = QuadraticProblem(mdp, 'a', 'b', 'pos', debug=True)
    r = qp.solve_lower_upper().df()
    print(r)
    assert (r['lower_reachability_value'] == r['upper_reachability_value']).all()
    assert r['lower_reachability_value'].iloc[0] == 0.5, r['lower_reachability_value'].iloc[0]
    assert (r['lower_importance_value'] == r['upper_importance_value']).all()
    assert r['lower_importance_value'].iloc[0] == 1, r['lower_importance_value'].iloc[0]
    return (r)

def paper_example():
    mdp = paper_example_mdp()

    qp = QuadraticProblem(mdp, 'q0: start_customer', 'consultation_customer', 'positive', debug=True)
    # qp = QuadraticProblem(mdp, 'q0: start_customer', 'angry', 'positive', debug=True)
    
    r = qp.solve_lower_upper().df()
    print(r)
    assert (r['lower_reachability_value'] == r['upper_reachability_value']).all()
    assert r['lower_reachability_value'].iloc[0] == 0.98, r['lower_reachability_value'].iloc[0]
    assert round(r['lower_importance_value'].iloc[0], 3) == 0.525, round(r['lower_importance_value'].iloc[0], 3)
    assert r['upper_importance_value'].iloc[0] == 1, r['lower_importance_value'].iloc[0]
    return r

if __name__ == '__main__':   
    
    # epidemic_influence_example()
    # assert False
    # loop_example()
    # assert False
    # gridworld_experiment()
    # assert False
    # G = nx.MultiDiGraph()
    # G.add_edges_from([('0','1'),('1', '3',), ('0', '2'), ('2', '3')])
    # print(G)
    # print(unroll(G, '1', '0').nodes)
    
    loop_example_half()
    assert False
    
    # paper_example()
    # assert False
    
    mdp = nx.MultiDiGraph()
    mdp.add_edge('s0', 'st', action = 'a', prob_weight = 1)
    mdp.add_edge('s0', 'st', action = 'b', prob_weight = 0.1)
    mdp.add_edge('s0', 's1', action = 'b', prob_weight = 0.8)
    mdp.add_edge('s0', 'sink', action = 'b', prob_weight = 0.1)
    mdp.add_edge('s1', 'st', action = 'a', prob_weight = 1)
    mdp.add_edge('s1', 'st', action = 'b', prob_weight = 0.1)
    mdp.add_edge('s1', 's2', action = 'b', prob_weight = 0.8)
    mdp.add_edge('s1', 'sink', action = 'b', prob_weight = 0.1)
    mdp.add_edge('s2', 'st', action = 'b', prob_weight = 0.1)
    mdp.add_edge('s2', 'sink', action = 'b', prob_weight = 0.9)
    print('prior', [(e, mdp.edges[e]) for e in mdp.edges])
    
    unrolled = unroll(add_self_loops(mdp), 's1', 's0')
    print(unrolled.nodes)
    for e in unrolled.edges:
        print(e, unrolled.edges[e])
    
    qp = QuadraticProblem(mdp, 's0', 's2', 'st', debug=True)
    print(qp.solve_lower_upper().df())

    
# TODO: Can actions being binary be further exploited?
# TODO: Do I need all 4 cases in problem?
# TODO assert that first objective is optimal for reachability
# TODO rewrite into one set of variables p_s
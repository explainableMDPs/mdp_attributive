import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from Result import GurobiResult, GurobiResultLowerUpper
import random
random.seed(42)

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
    
class QuadraticProblem:
    
    def __init__(self, model : nx.MultiDiGraph, start_state : str, via_state, target_state : str, timeout = 10*60*60, threads = 5, debug = False, memory=20):   
        self.env = gp.Env()
        self.m = gp.Model("qp", env=self.env)
        self.m.setParam('TimeLimit', timeout)
        self.m.setParam('SoftMemLimit', memory)
        self.m.setParam('Threads', threads)
        
        self.model = unroll(add_self_loops(model), via_state, start_state)
        self.start_state = start_state
        self.via_state= via_state
        self.target_state = target_state
        self.timeout = timeout
        self.debug = debug

        assert start_state in model        
        assert target_state in model
        assert via_state in model 
                
        print('Nodes', self.model.nodes)
        print()
        
        self.p_s_t = {s : self.m.addVar(ub=1.0, name=f'p_{str(s)}->t', lb = 0) for s in self.model.nodes}
        self.p_s_f = {s : self.m.addVar(ub=1.0, name=f'p_{str(s)}->f', lb = 0) for s in self.model.nodes}
        self.p_sa = {}

        # start_state = [s for s in self.model.nodes if 'q0: start' in s]
        # assert len(start_state) == 1, start_state
        # self.start_state = start_state[0]
        
        # target_state = [s for s in self.model.nodes if 'negative' in s]
        # assert len(target_state) == 1, target_state
        # self.target_state = target_state[0]
        if (self.target_state, 'f') not in self.model.nodes:
            self.reaching_states = [s for s in self.model.nodes if s[0] != self.target_state and nx.has_path(self.model, s, (self.target_state, 't'))]
            self.m.addConstr(self.p_s_t[(self.target_state, 't')] == 1)
            self.m.addConstr(self.p_s_f[(self.target_state, 't')] == 0)
        elif (self.target_state, 't') not in self.model.nodes:
            self.reaching_states = [s for s in self.model.nodes if s[0] != self.target_state and nx.has_path(self.model, s, (self.target_state, 'f'))]
            print("no pos", self.via_state)
            self.m.addConstr(self.p_s_t[(self.target_state, 'f')] == 0)
            self.m.addConstr(self.p_s_f[(self.target_state, 'f')] == 1)
        else:
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
                
    def get_max_solution(self):
        if self.debug:
            print(f"Found {self.m.SolCount} solutions:")
        solutions = []
        for s in range(self.m.SolCount):
            # Set which solution we will query from now on
            self.m.params.SolutionNumber = s
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
        return max(solutions)
        
    def get_solution(self):
        if self.m.status == GRB.INFEASIBLE:
            return GurobiResult(time=self.m.Runtime, reachability_value=-0.2, importance_value=-0.2, start_state=self.start_state, via_state=self.via_state, target_state=self.target_state, timeout=self.timeout, status=self.m.status)
        
        # compute result as in diverse target function include determinant
        if self.m.status == GRB.TIME_LIMIT:
            if self.m.SolCount == 0:
                return GurobiResult(time=self.m.Runtime, reachability_value=0, importance_value=0, start_state=self.start_state, via_state=self.via_state, target_state=self.target_state, timeout=self.timeout, status=self.m.status)
            else:
                max_result = self.get_max_solution()
                return GurobiResult(time=self.m.Runtime, reachability_value=max_result[0], importance_value=max_result[1], start_state=self.start_state, via_state=self.via_state, target_state=self.target_state, timeout=self.timeout, status=self.m.status)
        max_result = self.get_max_solution() 
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
            
        if sense == GRB.MAXIMIZE:
            self.m.setObjectiveN(self.goal_var, index = 1, priority=0)
        else:
            self.m.setObjectiveN(-self.goal_var, index = 1, priority=0)
            
        self.m.ModelSense = GRB.MAXIMIZE
        # self.m.setObjective(self.goal_var, sense = sense)
        
        self.m.update()
        
        self.m.optimize()
        # print(self.m.display())
        
        assert self.p_s_t[(self.start_state, 'f')].X + self.p_s_f[(self.start_state, 'f')].X != 0, f'Denominator is valued at 0'
        
        return_result = self.get_solution()
        if self.m.status == GRB.INFEASIBLE or self.m.status == GRB.TIME_LIMIT:
            self.m.dispose()
            return return_result
        
        assert self.m.status == GRB.OPTIMAL, f'Status is {self.m.status}'
        print("Importance")
        print("goal_var", self.goal_var.X)
        if self.debug:
            for v in self.m.getVars():
                print(f"{v.VarName} {v.X:g}")
            print(f"Obj: {self.m.ObjVal:g}")
        
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
    
if __name__ == '__main__':
    print("Test")
    
    # G = nx.MultiDiGraph()
    # G.add_edges_from([('0','1'),('1', '3',), ('0', '2'), ('2', '3')])
    # print(G)
    # print(unroll(G, '1', '0').nodes)
    
    
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
import numpy as np
import networkx as nx
from scipy.stats import hypergeom


def epidemic_influence_mdp(max_pop, debug = False) -> nx.MultiDiGraph:
    """Function to generate epidemic influence MDP from 
        https://github.com/ddv-lab/counterfactual-influence-in-MDPs/blob/main/experiments/epidemic_influence_experiments.ipynb
        Note that normalization on the transition probabilities was added.

    Returns:
        MDP : nx.MultiDiGraph
    """
    # Constants
    MAX_POPULATION = max_pop

    # S: Susceptible to the disease
    # I: Infected
    # V: Vaccines available
    state_space = [(S, I, V) for S in range(MAX_POPULATION + 1)
                            for I in range(MAX_POPULATION + 1)
                            for V in range(2 * MAX_POPULATION + 1)]
    num_states = len(state_space)
    state_index = {state: i for i, state in enumerate(state_space)}
    state_from_index = {i: state for i, state in enumerate(state_space)}

    # Action Space
    actions = ["NIL", "V_I", "V_S"]
    action_index = {"NIL": 0, "V_I": 1, "V_S": 2}
    action_from_index = {0: "NIL", 1: "V_I", 2: "V_S"}
    num_actions = len(actions)
    
    # Initialize Transition Matrix and Reward Matrix
    transition_matrix = np.zeros((num_actions, num_states, num_states))
    reward_matrix = np.zeros((num_states, num_actions))
    
    # Function to compute transition probabilities
    def compute_transitions(S, I, V, action):
        M = S + I  # Total population for hypergeometric distribution
        transitions = {}

        if action == "NIL":
            N = S
            n = min(S, I)
            V_prime = V

            for k in range(S + 1):
                prob = hypergeom(M, n, N).pmf(k)
                S_prime, I_prime = S - k, I + k

                if S_prime >= 0 and I_prime <= MAX_POPULATION:
                    transitions[(S_prime, I_prime, V_prime)] = prob

        elif action == "V_I" and I > 0 and V > 0:
            M -= 1
            N = S
            n = min(S, I - 1)
            V_prime = V - 1

            for k in range(S + 1):
                prob = hypergeom(M, n, N).pmf(k)
                if S == 0:
                    prob = 1
                S_prime, I_prime = S - k, I - 1 + k
                if S_prime >= 0 and I_prime <= MAX_POPULATION:
                    transitions[(S_prime, I_prime, V_prime)] = prob

        elif action == "V_S" and S > 0 and V > 0:
            M -= 1
            N = S - 1
            n = min(S - 1, I)
            V_prime = V - 1
            for k in range(S + 1):
                prob = hypergeom(M, n, N).pmf(k)
                if S == 1 and k == 0:
                    prob = 1
                S_prime, I_prime = S - 1 - k, I + k
                if S_prime >= 0 and I_prime <= MAX_POPULATION:
                    transitions[(S_prime, I_prime, V_prime)] = prob

        return transitions

    # Compute Transition and Reward Matrices
    for action_idx, action in enumerate(actions):
        for state in state_space:
            S, I, V = state
            state_idx = state_index[state]
            transitions = compute_transitions(S, I, V, action)
            # Update transition matrix
            for next_state, prob in transitions.items():
                next_state_idx = state_index[next_state]
                transition_matrix[action_idx, state_idx, next_state_idx] = prob

            # Update reward matrix (negative of the number of infected individuals)
            reward_matrix[state_idx, action_idx] = -I

    transition_matrix = np.nan_to_num(transition_matrix)
    # Output the size of the matrices for verification
    print("MDP size:")
    print(transition_matrix.shape)
    print(reward_matrix.shape)
    
    precision = 10
    mdp = nx.MultiDiGraph()
    mdp.add_nodes_from(state_space)
    for s in state_space:
        if debug:
            print("s", s)
        for action_idx, action in enumerate(actions):
            if debug:
                print("--- action", action)
            added_sum = 0
            sum_probabilities = sum(round(transition_matrix[action_idx,state_index[s],state_index[t]], precision) for t in state_space)
            if sum_probabilities == 0:
                continue
            for t in state_space:
                p = transition_matrix[action_idx,state_index[s],state_index[t]]
                if round(p, precision) == 0 and p != 0:
                    assert False, f'Rounded probability to 0:{p}'
                p = round(p, precision) / sum_probabilities
                if debug and p != 0:
                    print("--- --- t", t, ':', p)
                if p != 0:
                    mdp.add_edge(s, t, action = action, prob_weight = p)
                    if debug:
                        print(f'--- --- Added {s} -{action}-> {t} : {p} ({transition_matrix[action_idx,state_index[s],state_index[t]]})')
                    added_sum += p
            assert round(added_sum, precision) == 1 or added_sum == 0, f'{added_sum}, {transition_matrix[action_idx,state_index[(1,1,0)],state_index[(1,1,0)]]}'
    # Define winning state
    # If no one can be infected, game is won
    for V in range(2 * MAX_POPULATION + 1):
        mdp.add_edge('q0: start', (MAX_POPULATION, 1, V),  action = f'vaccination_{V}', prob_weight = 1)
        mdp.add_edge((0,0,V), 'positive',  action = 'won', prob_weight = 1)
        # for I in range(MAX_POPULATION + 1):
            # mdp.add_edge((I,0,V), 'positive',  action = 'won', prob_weight = 1)
            
    return mdp

def gridworld_mdp():
    mdp = nx.MultiDiGraph()
    nodes = [f's{i}{j}' for i in range(0, 7) for j in range(0, 7)]
    transitions_r = [(f's{i}{j}', f's{min(i+1, 6)}{j}') for i in [0, 1, 2, 4, 5, 6] for j in range(0,7) if i+1 <= 6]
    transitions_l = [(f's{i}{j}', f's{max(i-1, 0)}{j}') for i in [0, 1, 2, 4, 5, 6] for j in range(0,7) if i-1 >= 0]
    transitions_u = [(f's{i}{j}', f's{i}{min(j+1, 6)}') for i in [0, 1, 2, 4, 5, 6] for j in range(0,7) if j+1 <= 6]
    transitions_d = [(f's{i}{j}', f's{i}{max(j-1, 0)}') for i in [0, 1, 2, 4, 5, 6] for j in range(0,7) if j-1 >= 0]
    transitions_caught = [(f's{i}{j}',f's{i}{j}') for i in [3] for j in [0, 1, 2, 4, 5, 6]]
    # encode (3,3) position
    transitions_3r = [(f's{i}{j}', f's{min(i+1, 6)}{j}') for i in [3] for j in [3]]
    transitions_3l = [(f's{i}{j}', f's{max(i-1, 0)}{j}') for i in [3] for j in [3]]
    transitions_3d = [(f's{i}{j}', f's{i}{min(j+1, 6)}') for i in [3] for j in [3]]
    transitions_3u = [(f's{i}{j}', f's{i}{max(j-1, 0)}') for i in [3] for j in [3]]
    
    mdp.add_nodes_from(nodes)
    mdp.add_edges_from(transitions_r, action = 'r', prob_weight=1)
    mdp.add_edges_from(transitions_l, action = 'l', prob_weight=1)
    mdp.add_edges_from(transitions_d, action = 'd', prob_weight=1)
    mdp.add_edges_from(transitions_u, action = 'u', prob_weight=1)
    # mdp.add_edges_from(transitions_caught, action= 'c',  prob_weight=1)
    mdp.add_edges_from(transitions_3r, action = 'r', prob_weight=1)
    mdp.add_edges_from(transitions_3l, action = 'l', prob_weight=1)
    mdp.add_edges_from(transitions_3d, action = 'd', prob_weight=1)
    mdp.add_edges_from(transitions_3u, action = 'u', prob_weight=1)
    
    return mdp

def gridworld_mdp_with_key():
    mdp = nx.MultiDiGraph()
    nodes = [f's{i}{j}{key}' for i in range(0, 7) for j in range(0, 7) for key in [True, False]]
    transitions_r = [(f's{i}{j}{key}', f's{min(i+1, 6)}{j}{key}') for i in [0, 1, 2, 4, 5, 6] for j in range(0,7) if i+1 <= 6 for key in [True, False]]
    transitions_l = [(f's{i}{j}{key}', f's{max(i-1, 0)}{j}{key}') for i in [0, 1, 2, 4, 5, 6] for j in range(0,7) if i-1 >= 0 for key in [True, False]]
    transitions_u = [(f's{i}{j}{key}', f's{i}{min(j+1, 6)}{key}') for i in [0, 1, 2, 4, 5, 6] for j in range(0,7) if j+1 <= 6 for key in [True, False]]
    transitions_d = [(f's{i}{j}{key}', f's{i}{max(j-1, 0)}{key}') for i in [0, 1, 2, 4, 5, 6] for j in range(0,7) if j-1 >= 0 for key in [True, False]]
    transitions_caught = [(f's{i}{j}',f's{i}{j}') for i in [3] for j in [0, 1, 2, 4, 5, 6]]
    # encode (3,3) position
    transitions_3r = [(f's{i}{j}{key}', f's{min(i+1, 6)}{j}{key}') for i in [3] for j in [3] for key in [True, False]]
    transitions_3l = [(f's{i}{j}{key}', f's{max(i-1, 0)}{j}{key}') for i in [3] for j in [3] for key in [True, False]]
    transitions_3d = [(f's{i}{j}{key}', f's{i}{min(j+1, 6)}{key}') for i in [3] for j in [3] for key in [True, False]]
    transitions_3u = [(f's{i}{j}{key}', f's{i}{max(j-1, 0)}{key}') for i in [3] for j in [3] for key in [True, False]]
    
    mdp.add_nodes_from(nodes)
    mdp.add_edges_from(transitions_r, action = 'r', prob_weight=1)
    mdp.add_edges_from(transitions_l, action = 'l', prob_weight=1)
    mdp.add_edges_from(transitions_d, action = 'd', prob_weight=1)
    mdp.add_edges_from(transitions_u, action = 'u', prob_weight=1)
    # mdp.add_edges_from(transitions_caught, action= 'c',  prob_weight=1)
    mdp.add_edges_from(transitions_3r, action = 'r', prob_weight=1)
    mdp.add_edges_from(transitions_3l, action = 'l', prob_weight=1)
    mdp.add_edges_from(transitions_3d, action = 'd', prob_weight=1)
    mdp.add_edges_from(transitions_3u, action = 'u', prob_weight=1)
    
    mdp.add_edge('s00False', 's00True', action='pickup', prob_weight=1)
    mdp.remove_edge('s23False', 's33False') # no transition without key
    return mdp

def loop_mdp():
    # d <- a (<-> b) -> c
    mdp = nx.MultiDiGraph()
    mdp.add_edge('a', 'b', action = 'to_b', prob_weight = 1)
    mdp.add_edge('a', 'd', action = 'to_d', prob_weight = 1)
    mdp.add_edge('a', 'c', action = 'to_c', prob_weight = 1)
    mdp.add_edge('b', 'a', action = 'to_a', prob_weight = 1)
    mdp.add_edge('b', 'c', action = 'to_c', prob_weight = 1)
    return mdp

def loop_half_mdp():
    # a -> b <-> c -> d -> 0.5 : pos + 0.5 : neg
    mdp = nx.MultiDiGraph()
    mdp.add_edge('a', 'b', action = 'to_b', prob_weight = 1)
    mdp.add_edge('b', 'c', action = 'to_c', prob_weight = 1)
    mdp.add_edge('c', 'b', action = 'to_b', prob_weight = 1)
    mdp.add_edge('c', 'd', action = 'to_d', prob_weight = 1)
    mdp.add_edge('d', 'pos', action = 'to_end', prob_weight = 0.5)
    mdp.add_edge('d', 'neg', action = 'to_end', prob_weight = 0.5)
    return mdp

def paper_example_mdp():
    mdp = nx.MultiDiGraph()
    mdp.add_edge("q0: start_customer", "error_customer", action = 'apply', prob_weight = 0.05, controllable=False)
    mdp.add_edge("q0: start_customer", "application", action = 'apply', prob_weight = 0.95, controllable=False)
    mdp.add_edge("q0: start_customer", "consultation_customer", action = 'consult', prob_weight = 1, controllable=False)

    mdp.add_edge("application", "consultation_customer", action = 'provider', prob_weight = 0.5, controllable=True)
    mdp.add_edge("application", "application+", action = 'provider', prob_weight = 0.5, controllable=True)

    mdp.add_edge("error_customer", "consultation_customer", action = 'consult', prob_weight = 1, controllable=False)
    mdp.add_edge("error_customer", "negative", action = 'quit', prob_weight = 1, controllable=False)

    mdp.add_edge("consultation_customer", "application+", action = 'apply', prob_weight = 1, controllable=False)
    mdp.add_edge("consultation_customer", "angry", action = 'quit', prob_weight = 1, controllable=False)

    mdp.add_edge("angry", "negative", action = 'quit', prob_weight = 1, controllable=False)

    mdp.add_edge("rework_customer", "resubmit", action = 'submit', prob_weight = 1, controllable=False)
    mdp.add_edge("rework_customer", "negative", action = 'quit', prob_weight = 1, controllable=False)

    mdp.add_edge("resubmit", "positive", action = 'provider', prob_weight = 0.8, controllable=True)
    mdp.add_edge("resubmit", "negative", action = 'provider', prob_weight = 0.2, controllable=True)

    mdp.add_edge("application+", "positive", action = 'provider', prob_weight = 0.9, controllable=True)
    mdp.add_edge("application+", "rework_customer", action = 'provider', prob_weight = 0.1, controllable=True)
    return mdp

if __name__ == '__main__':
    g = epidemic_influence_mdp(debug=True)
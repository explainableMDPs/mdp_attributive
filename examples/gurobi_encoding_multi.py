import gurobipy as gp
from gurobipy import GRB

def problem_1():
    # Create a new model
    m = gp.Model("qp")
    #m.Params.NonConvex = 0

    # places = [m.addVar(ub=1.0, name='p%s' % i, lb = 0) for i in range(9)]
    placesP0 = [m.addVar(ub=1.0, name='p0_P0', lb = 0),m.addVar(ub=1.0, name='p1_P0', lb = 0),m.addVar(ub=1.0, name='p2_P0', lb = 0),
            m.addVar(ub=1.0, name='ptT_P0', lb = 0),m.addVar(ub=1.0, name='ptB_P0', lb = 0),m.addVar(ub=1.0, name='sink_P0', lb = 0)]

    placesP2 = [m.addVar(ub=1.0, name='p0_P2', lb = 0),m.addVar(ub=1.0, name='p1_P2', lb = 0),m.addVar(ub=1.0, name='p2_P2', lb = 0),
            m.addVar(ub=1.0, name='ptT_P2', lb = 0),m.addVar(ub=1.0, name='ptB_P2', lb = 0),m.addVar(ub=1.0, name='sink_P2', lb = 0)]

    placesP1 = [m.addVar(ub=1.0, name='p0_P1', lb = 0),m.addVar(ub=1.0, name='p1_P1', lb = 0),m.addVar(ub=1.0, name='p2_P1', lb = 0),
            m.addVar(ub=1.0, name='ptT_P1', lb = 0),m.addVar(ub=1.0, name='ptB_P1', lb = 0),m.addVar(ub=1.0, name='sink_P1', lb = 0)]

    p0_a = m.addVar(ub=1.0, name='p0_a', lb = 0)
    p0_b = m.addVar(ub=1.0, name='p0_b', lb = 0)
    p1_a = m.addVar(ub=1.0, name='p1_a', lb = 0)
    p1_b = m.addVar(ub=1.0, name='p1_b', lb = 0)
    p2_b = m.addVar(ub=1.0, name='p2_b', lb = 0)

    all_vars = [p0_a, p0_b, p1_a, p1_b, p2_b]

    for v in all_vars:
        m.addConstr(v <= 1)
        #m.addConstr(v >= 0)


    m.addConstr(p0_a + p0_b == 1)
    m.addConstr(p1_a + p1_b == 1)
    m.addConstr(p2_b == 1)

    m.addConstr(placesP0[-1] == 0) # sink receives 0
    m.addConstr(placesP0[-2] == 1) # Bottom endstate receives 1
    m.addConstr(placesP0[-3] == 1) # Top endstate receives 1 - general reachability
    m.addConstr(placesP0[2] == p2_b * (0.1 * placesP0[-3] + 0.9 * placesP0[-1]))
    m.addConstr(placesP0[1] == p1_b * (0.1 * placesP0[-2] + 0.8 * placesP0[2] + 0.1 * placesP0[-1]) + p1_a * placesP0[-3])
    m.addConstr(placesP0[0] == p0_b * (0.1 * placesP0[-2] + 0.8 * placesP0[1] + 0.1 * placesP0[-1]) + p0_a * placesP0[-2])

    m.addConstr(placesP1[-1] == 0) # sink receives 0
    m.addConstr(placesP1[-2] == 1) # Bottom endstate receives 0
    m.addConstr(placesP1[-3] == 1) # Top endstate receives 1
    m.addConstr(placesP1[2] == p2_b * (0.1 * placesP1[-3] + 0.9 * placesP1[-1]))
    m.addConstr(placesP1[1] == p1_b * (0.1 * placesP1[-3] + 0.8 * placesP1[2] + 0.1 * placesP1[-1]) + p1_a * 1)
    m.addConstr(placesP1[0] == p0_b * (0.1 * placesP1[-2] + 0.8 * placesP1[1] + 0.1 * placesP1[-1]) + p0_a * 0)

    m.addConstr(placesP2[-1] == 0) # sink receives 0
    m.addConstr(placesP2[-2] == 0) # Bottom endstate receives 0
    m.addConstr(placesP2[-3] == 1) # Top endstate receives 1
    m.addConstr(placesP2[2] == p2_b * (0.1 * placesP2[-3] + 0.9 * placesP2[-1]))
    m.addConstr(placesP2[1] == p1_b * (0.1 * placesP2[-2] + 0.8 * placesP2[2] + 0.1 * placesP2[-1]) + p1_a * placesP2[-2])
    m.addConstr(placesP2[0] == p0_b * (0.1 * placesP2[-2] + 0.8 * placesP2[1] + 0.1 * placesP2[-1]) + p0_a * placesP2[-2])

    # t = m.addVar(name='target_fraction', lb = 0, ub=1)

    max_var = m.addVar(name='max', lb = 0, ub=1)
    m.addConstr(max_var * placesP0[0] <= placesP1[0])
    m.addConstr(max_var * placesP0[0] <= placesP2[0])

    # m.addConstr(t * placesP0[0] == placesP1[0])
    # m.addConstr(placesP2[0] + placesP1[0] >= 0.00000001)

    # Set objective
    # obj = t
    # IMPORTANT: For test purposes, maximize relevance of states
    # GUROBI finds more than one solution TODO
    m.setObjective(placesP0[0], sense = GRB.MAXIMIZE)
    m.setObjectiveN(max_var, index = 1, priority=1)

    m.optimize()

    for v in m.getVars():
        print(f"{v.VarName} {v.X:g}")

    print(f"Obj: {m.ObjVal:g}")

def problem_2():
        # Create a new model
    m = gp.Model("qp")
    #m.Params.NonConvex = 0

    # places = [m.addVar(ub=1.0, name='p%s' % i, lb = 0) for i in range(9)]
    placesP0 = [m.addVar(ub=1.0, name='p0_P0', lb = 0),m.addVar(ub=1.0, name='p1_P0', lb = 0),m.addVar(ub=1.0, name='p2_P0', lb = 0),
            m.addVar(ub=1.0, name='ptT_P0', lb = 0),m.addVar(ub=1.0, name='ptB_P0', lb = 0)]

    placesP2 = [m.addVar(ub=1.0, name='p0_P2', lb = 0),m.addVar(ub=1.0, name='p1_P2', lb = 0),m.addVar(ub=1.0, name='p2_P2', lb = 0),
            m.addVar(ub=1.0, name='ptT_P2', lb = 0),m.addVar(ub=1.0, name='ptB_P2', lb = 0)]

    placesP1 = [m.addVar(ub=1.0, name='p0_P1', lb = 0),m.addVar(ub=1.0, name='p1_P1', lb = 0),m.addVar(ub=1.0, name='p2_P1', lb = 0),
            m.addVar(ub=1.0, name='ptT_P1', lb = 0),m.addVar(ub=1.0, name='ptB_P1', lb = 0)]

    p0_a = m.addVar(ub=1.0, name='p0_a', lb = 0)
    p0_b = m.addVar(ub=1.0, name='p0_b', lb = 0)
    p1_a = m.addVar(ub=1.0, name='p1_a', lb = 0)
    p1_b = m.addVar(ub=1.0, name='p1_b', lb = 0)
    p2_a = m.addVar(ub=1.0, name='p2_a', lb = 0)
    p2_b = m.addVar(ub=1.0, name='p2_b', lb = 0)

    all_vars = [p0_a, p0_b, p1_a, p1_b, p2_b]

    for v in all_vars:
        m.addConstr(v <= 1)
        #m.addConstr(v >= 0)


    m.addConstr(p0_a + p0_b == 1)
    m.addConstr(p1_a + p1_b == 1)
    m.addConstr(p2_b == 1)

    m.addConstr(placesP0[-1] == 0) # sink receives 0
    m.addConstr(placesP0[-2] == 1) # Bottom endstate receives 1
    m.addConstr(placesP0[-3] == 1) # Top endstate receives 1 - general reachability
    m.addConstr(placesP0[2] == p2_b * (0.1 * placesP0[-3] + 0.9 * placesP0[-1]))
    m.addConstr(placesP0[1] == p1_b * (0.1 * placesP0[-2] + 0.8 * placesP0[2] + 0.1 * placesP0[-1]) + p1_a * placesP0[-3])
    m.addConstr(placesP0[0] == p0_b * placesP0[1] + p0_a * placesP0[-2])

    m.addConstr(placesP1[-1] == 0) # sink receives 0
    m.addConstr(placesP1[-2] == 1) # Bottom endstate receives 0
    m.addConstr(placesP1[-3] == 1) # Top endstate receives 1
    m.addConstr(placesP1[2] == p2_b * (0.1 * placesP1[-3] + 0.9 * placesP1[-1]))
    m.addConstr(placesP1[1] == p1_b * (0.1 * placesP1[-3] + 0.8 * placesP1[2] + 0.1 * placesP1[-1]) + p1_a * 1)
    m.addConstr(placesP1[0] == p0_b * (0.1 * placesP1[-2] + 0.8 * placesP1[1] + 0.1 * placesP1[-1]) + p0_a * 0)

    m.addConstr(placesP2[-1] == 0) # sink receives 0
    m.addConstr(placesP2[-2] == 0) # Bottom endstate receives 0
    m.addConstr(placesP2[-3] == 1) # Top endstate receives 1
    m.addConstr(placesP2[2] == p2_b * (0.1 * placesP2[-3] + 0.9 * placesP2[-1]))
    m.addConstr(placesP2[1] == p1_b * (0.1 * placesP2[-2] + 0.8 * placesP2[2] + 0.1 * placesP2[-1]) + p1_a * placesP2[-2])
    m.addConstr(placesP2[0] == p0_b * (0.1 * placesP2[-2] + 0.8 * placesP2[1] + 0.1 * placesP2[-1]) + p0_a * placesP2[-2])

    # t = m.addVar(name='target_fraction', lb = 0, ub=1)

    max_var = m.addVar(name='max', lb = 0, ub=1)
    m.addConstr(max_var * placesP0[0] <= placesP1[0])
    m.addConstr(max_var * placesP0[0] <= placesP2[0])

    # m.addConstr(t * placesP0[0] == placesP1[0])
    # m.addConstr(placesP2[0] + placesP1[0] >= 0.00000001)

    # Set objective
    # obj = t
    # IMPORTANT: For test purposes, maximize relevance of states
    # GUROBI finds more than one solution TODO
    m.setObjective(placesP0[0], sense = GRB.MAXIMIZE)
    m.setObjectiveN(max_var, index = 1, priority=1)

    m.optimize()

    for v in m.getVars():
        print(f"{v.VarName} {v.X:g}")

    print(f"Obj: {m.ObjVal:g}")
    
# problem_1()
problem_2()
#x.VType = GRB.INTEGER
#y.VType = GRB.INTEGER
#z.VType = GRB.INTEGER
#    
#m.optimize()
#
#for v in m.getVars():
#    print(f"{v.VarName} {v.X:g}")
#
#print(f"Obj: {m.ObjVal:g}")
import gurobipy as gp
from gurobipy import GRB

# Create a new model
m = gp.Model("qp")
#m.Params.NonConvex = 0

# places = [m.addVar(ub=1.0, name='p%s' % i, lb = 0) for i in range(9)]
placesB = [m.addVar(ub=1.0, name='p0_B', lb = 0),m.addVar(ub=1.0, name='p1_B', lb = 0),m.addVar(ub=1.0, name='p2_B', lb = 0),
          m.addVar(ub=1.0, name='ptT_B', lb = 0),m.addVar(ub=1.0, name='ptB_B', lb = 0),m.addVar(ub=1.0, name='sink_B', lb = 0)]

placesT = [m.addVar(ub=1.0, name='p0_T', lb = 0),m.addVar(ub=1.0, name='p1_T', lb = 0),m.addVar(ub=1.0, name='p2_T', lb = 0),
          m.addVar(ub=1.0, name='ptT_T', lb = 0),m.addVar(ub=1.0, name='ptB_T', lb = 0),m.addVar(ub=1.0, name='sink_T', lb = 0)]

p0_a = m.addVar(ub=1.0, name='p0_a', lb = 0, vtype=gp.GRB.BINARY)
p0_b = m.addVar(ub=1.0, name='p0_b', lb = 0, vtype=gp.GRB.BINARY)
p1_a = m.addVar(ub=1.0, name='p1_a', lb = 0, vtype=gp.GRB.BINARY)
p1_b = m.addVar(ub=1.0, name='p1_b', lb = 0, vtype=gp.GRB.BINARY)
p2_b = m.addVar(ub=1.0, name='p2_b', lb = 0, vtype=gp.GRB.BINARY)

all_vars = [p0_a, p0_b, p1_a, p1_b, p2_b]

for v in all_vars:
    m.addConstr(v <= 1)
    #m.addConstr(v >= 0)


m.addConstr(p0_a + p0_b == 1)
m.addConstr(p1_a + p1_b == 1)
m.addConstr(p2_b == 1)

m.addConstr(placesB[-1] == 0) # sink receives 0
m.addConstr(placesB[-2] == 1) # Bottom endstate receives 1
m.addConstr(placesB[-3] == 0) # Top endstate receives 0
m.addConstr(placesB[2] == p2_b * (0.1 * placesB[-3] + 0.9 * placesB[-1]))
m.addConstr(placesB[1] == p1_b * (0.1 * placesB[-2] + 0.8 * placesB[2] + 0.1 * placesB[-1]) + p1_a * placesB[-2])
m.addConstr(placesB[0] == p0_b * (0.1 * placesB[-2] + 0.8 * placesB[1] + 0.1 * placesB[-1]) + p0_a * placesB[-2])

m.addConstr(placesT[-1] == 0) # sink receives 0
m.addConstr(placesT[-2] == 0) # Bottom endstate receives 0
m.addConstr(placesT[-3] == 1) # Top endstate receives 1
m.addConstr(placesT[2] == p2_b * (0.1 * placesT[-3] + 0.9 * placesT[-1]))
m.addConstr(placesT[1] == p1_b * (0.1 * placesT[-2] + 0.8 * placesT[2] + 0.1 * placesT[-1]) + p1_a * placesT[-2])
m.addConstr(placesT[0] == p0_b * (0.1 * placesT[-2] + 0.8 * placesT[1] + 0.1 * placesT[-1]) + p0_a * placesT[-2])

t = m.addVar(name='target_fraction', lb = 0, ub=1)

m.addConstr(t * (placesT[0] + placesB[0]) == placesT[0])
m.addConstr(placesT[0] + placesB[0] >= 0.001)

# Set objective
obj = t
#see https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.setObjectiveN
m.setObjectiveN(placesT[0] + placesB[0], index = 0, priority = 1)
m.setObjectiveN(obj, index = 1, priority=0)
m.ModelSense = GRB.MAXIMIZE

m.optimize()

for v in m.getVars():
    print(f"{v.VarName} {v.X:g}")

print(f"Reach: {placesT[0].X + placesB[0].X}")
print(f"Importance: {t.X:g}")

nSolutions  = m.SolCount
nObjectives = m.NumObj
print('Problem has', nObjectives, 'objectives')
print('Gurobi found', nSolutions, 'solutions')
solutions = []
for s in range(nSolutions):
    # Set which solution we will query from now on
    m.params.SolutionNumber = s
    print('Solution', s, ':', end='')
    for o in range(nObjectives):
        # Set which objective we will query
        m.params.ObjNumber = o
        # Query the o-th objective value
        print(' ', m.ObjNVal, end='')
      # Print first three variables in the solution
    print('')


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
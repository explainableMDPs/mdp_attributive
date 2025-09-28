import pandas as pd

class GurobiResult:
    def __init__(self, time, reachability_value, importance_value, start_state, via_state, target_state, timeout, status = -1):
        self.time = time
        self.reachability_value = reachability_value
        self.importance_value = importance_value
        self.start_state = start_state
        self.via_state = via_state
        self.target_state = target_state
        self.timeout = timeout
        self.status = status # https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html
        
    def df(self):
        d = {'time' : self.time, 'reachability_value' : self.reachability_value, 'importance_value' : self.importance_value, 'start_state' : self.start_state, 'via_state' : self.via_state, 'target_state' : self.target_state, 
             'timeout' : self.timeout, 'status' : self.status}
        return pd.DataFrame([d])
    
class GurobiResultLowerUpper:
    def __init__(self, time, lower_reachability_value, lower_importance_value, upper_reachability_value, upper_importance_value, start_state, via_state, target_state, timeout, status = -1):
        self.time = time
        self.lower_reachability_value = lower_reachability_value
        self.lower_importance_value = lower_importance_value
        self.upper_reachability_value = upper_reachability_value
        self.upper_importance_value = upper_importance_value
        self.start_state = start_state
        self.via_state = via_state
        self.target_state = target_state
        self.timeout = timeout
        self.status = status # https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html
        
    def df(self):
        d = {'time' : self.time, 'lower_reachability_value' : self.lower_reachability_value,
             'lower_importance_value' : self.lower_importance_value,
             'upper_reachability_value' : self.upper_reachability_value,
             'upper_importance_value' : self.upper_importance_value,
             'start_state' : self.start_state, 'via_state' : self.via_state, 'target_state' : self.target_state, 
             'timeout' : self.timeout, 'status' : self.status}
        return pd.DataFrame([d])
    
class PrismResult():
    def __init__(self, model, query, nesting, value, model_parsing_time, model_checking_time, model_construction_time, name, timeout, solver):
        self.model = model
        self.query = query
        self.nesting = nesting
        self.value = value
        self.model_parsing_time = model_parsing_time
        self.model_checking_time = model_checking_time
        self.model_construction_time = model_construction_time
        self.name = name
        self.timeout = timeout
        self.solver = solver
        
    def df(self):
        d = {'model':self.model, 
             'query':self.query, 
             'nesting':self.nesting,
             'value' : self.value, 
             'model_parsing_time':self.model_parsing_time,
             'model_checking_time':self.model_checking_time,
             'model_construction_time':self.model_construction_time,
             'name':self.name,
             'timeout':self.timeout,
             'solver': self.solver}
        return pd.DataFrame([d])
        
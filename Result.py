import pandas as pd

class GurobiResult:
    def __init__(self, time, value, start_state, via_state, target_state, timeout, gap, status = -1):
        self.time = time
        self.value = value
        self.start_state = start_state
        self.via_state = via_state
        self.target_state = target_state
        self.timeout = timeout
        self.gap = gap
        self.status = status # https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html
        
    def df(self):
        d = {'time' : self.time, 'value' : self.value, 'start_state' : self.start_state, 'via_state' : self.via_state, 'target_state' : self.target_state, 
             'timeout' : self.timeout, 'gap' : self.gap, 'status' : self.status}
        return pd.DataFrame([d])
    
class GurobiResultLowerUpper:
    def __init__(self, time, value_lower, value_upper, start_state, via_state, target_state, timeout, gap, status = -1):
        self.time = time
        self.value_lower = value_lower
        self.value_upper = value_upper
        self.start_state = start_state
        self.via_state = via_state
        self.target_state = target_state
        self.timeout = timeout
        self.gap = gap
        self.status = status # https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html
        
    def df(self):
        d = {'time' : self.time, 'value_lower' : self.value_lower, 'value_upper' : self.value_upper, 
             'start_state' : self.start_state, 'via_state' : self.via_state, 'target_state' : self.target_state, 
             'timeout' : self.timeout, 'gap' : self.gap, 'status' : self.status}
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
        
import pandas as pd

class Result:
    def __init__(self, time, value, target_prob, strategy, timeout, gap, status = -1):
        self.time = time
        self.value = value
        self.target_prob = target_prob
        self.strategy = strategy
        self.timeout = timeout
        self.gap = gap
        self.status = status # https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html
        
    def df(self):
        d = {'time' : self.time, 'value' : self.value, 'target_prob' : self.target_prob, 'timeout' : self.timeout, 'gap' : self.gap, 'status' : self.status}
        return pd.DataFrame([d])
    
class PrismResult():
    def __init__(self, value, model_checking_time, model_construction_time):
        self.value = value
        self.model_checking_time = model_checking_time
        self.model_construction_time = model_construction_time
    
    def df(self):
        d = {'value' : self.value, 'model_checking_time':self.model_checking_time, 'model_construction_time':self.model_construction_time}
        return pd.DataFrame([d])
        
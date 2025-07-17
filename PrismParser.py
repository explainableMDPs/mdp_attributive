import subprocess
from Result import PrismResult
import abc

class Parser:
    def __init__(self, path, model, timeout):
        self.path = path
        self.model= model
        self.timeout = timeout
        
    @staticmethod
    def parse_result(result, keyword):
        lines = result.stdout.split('\n')
        lines = [l for l in lines if keyword in l]
        assert len(lines) == 1, f'Got too many results, {lines} with keyword {keyword}, result was {result}'
        line = lines[0]
        print(line)
        line = line.replace(keyword, "").split(" ")[0].replace("s.", '')
        return line
    
    @abc.abstractmethod
    def call(self, query, name, args=""):
        return 
    
class PrismParser(Parser):
    def call(self, query, name, args=""):
        # result = subprocess.run([self.path, self.model, '-pf', 'Pmax=? [F "positive"]'], capture_output = True, text = True)
        result = subprocess.run([self.path, self.model, '-maxiters', '1000000', '-timeout', f'{self.timeout}', '-cuddmaxmem', '8g', '-pf', query], capture_output = True, text = True)
        assert not result.stderr      
        
        result_value = Parser.parse_result(result, 'Result: ')
        model_construction_time = float(self.parse_result(result, 'Time for model construction: '))
        model_checking_time = float(self.parse_result(result, 'Time for model checking: '))
        
        nesting = query.count("X")
        
        return PrismResult(model=self.model, query=query, nesting=nesting, value=result_value, model_parsing_time=0, model_checking_time=model_checking_time,
                           model_construction_time=model_construction_time, name=name, timeout=self.timeout, solver="prism").df()
        
class StormParser(Parser):
    def call(self, query, name, args=""):
        # result = subprocess.run([self.path, self.model, '-pf', 'Pmax=? [F "positive"]'], capture_output = True, text = True)
        result = subprocess.run([self.path, '--prism', self.model, '--prop', query, '--exact'], capture_output = True, text = True)
        assert not result.stderr      
        
        result_value = Parser.parse_result(result, 'Result (for initial states): ')
        model_parsing_time = float(self.parse_result(result, 'Time for model input parsing: '))
        model_construction_time = float(self.parse_result(result, 'Time for model construction: '))
        model_checking_time = float(self.parse_result(result, 'Time for model checking: '))
        
        nesting = query.count("X")
        
        return PrismResult(model=self.model, query=query, nesting=nesting, value=result_value, model_parsing_time=model_parsing_time, model_checking_time=model_checking_time,
                           model_construction_time=model_construction_time, name=name, timeout=self.timeout, solver='storm').df()
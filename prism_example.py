import pickle
import aalpy 

# from aalpy.utils import model_check_experiment, get_properties_file, get_correct_prop_values
# from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

# aalpy.paths.path_to_prism = "/home/paul/Downloads/prism-4.8.1-linux64-x86/bin/prism"
# aalpy.paths.path_to_properties = "/home/paul/Downloads"
    
# with open(f'/home/paul/Documents/mdp_ce/out/models/model_spotify9000_model-it_9.pickle', 'rb') as handle:
#     model = pickle.load(handle)
   
from aalpy.utils import mdp_2_prism_format
     
# # values, diff = model_check_experiment(get_properties_file('props.props'), get_correct_prop_values(example), model)



from LogParser import SpotifyParser
from LogParser import BPIC17AfterParser
# parser = BPIC17AfterParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
parser = SpotifyParser('data/spotify/', 'data/activities_spotify.xml', int(1000))
model = parser.build_benchmark()
print(type(model))
mdp_2_prism_format(mdp=model, name='mc_exp', output_path=f'out/mc_exp_1000.prism')

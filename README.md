# Attribution-based Explanations for Markov Decision Processes
This repository contains all files to reproduce the results reported in the paper `ttribution-based Explanations for Markov Decision Processes' submitted to IJCAI 2026.

# Artifact Structure
```
.
├── benchmarks.py
├── data
│   └── # Journey data
├── examples
│   ├── a2c_cartpole.zip
│   ├── gridworld_gym.ipynb
│   ├── gurobi_encoding_multi.py
│   ├── gurobi_encoding.py
│   ├── intro-example.prism
│   └── props.props
├── fixed_mdp.py
├── journepy
│   └── # Library for preprocessing
├── LogParser.py
├── main.py
├── modeltest.py
├── out
│   ├── models
│   │   └── # All computed Models
│   ├── results_epidemic.csv
│   ├── results_greps_bpic.csv
│   ├── results_spotify.csv
│   └── results_spotify_songs.csv
├── plots.ipynb # File to re-generate presented plots, even without running experiments
├── README.md
├── requirements.txt # Requirements to re-run file
├── Result.py
├── run.sh
└── solver.py
```

# Generating Attribution-based Explanations
To run the experiments, run the ```benchmarks.py``` script, individual experiments can directly be run by calling the solver in ```solver.py```.

The benchmark script offers several, among others:

```
  -t TIMEOUT, --timeout TIMEOUT
                        
  -sa SAMPLES, --samples SAMPLES Number of states to sample for attribution computation
  
  -c CORES, --cores CORES Cores to use to parallelize experiments
  
  -e EXPERIMENTS [EXPERIMENTS ...], --experiments EXPERIMENTS [EXPERIMENTS ...] Name of experiments to run
  
  -rm, --rebuild_models Rebuild models, implies rebuilding strategies
  
  -mi MODEL_ITERATIONS, --model_iterations MODEL_ITERATIONS Number of models to generate for each setting
  
  -as, --all_spotify    All spotify models in steps of 100 are generated
```
## Datasets

The used datasets are not contained in the artifact as their licensing prohibits sharing.
However, all models generated in our experiments are contained in ```out/models```. 
To generate models from scratch, please download the datasets and store them in ```data/```:

- [Greps](https://zenodo.org/records/10666884).
- [BPIC'12](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204/1).
- [BPIC'17](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884).
- [MSSD](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge).

The models for Epidemic can be generated as no dataset is required for their construction.

## Reproducing Results:
To reproduce the experiments, please ensure that a valid Gurobi key is accessible and all requirements from ```requirements.txt``` are installed.

The script ```run.sh``` contains commands to reproduce all results, the computation can be drastically reduced by changing the number of samples, and excluding an encoding from ```benchmarks.py```.

```
uv run benchmarks.py --experiments greps bpic12 bpic17-both --model_iterations 1 --iterations 1 --samples 1000 --timeout 3600 --cores 50 --rebuild_models
```

Rebuilds the models used for Greps, BPIC12, and BPIC17, and computes the importance of every state in these models.

# Plots
All results are contained in the ```out/results_*.csv``` files to investigate and reproduce the pltos without running the experiments.
The Jupyter Notebook [plots.ipynb](plots.ipynb) reproduces all tables and plots contained in the paper.
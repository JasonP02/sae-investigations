# %% Imports
%load_ext autoreload
%autoreload 2

from config import GenerationConfig
from setup import setup_model_and_sae
from experiment import run_multiple_experiments
from visualization import visualize_experiment_results
from models import ExperimentResults 
import torch
import gc
from tqdm import tqdm
import numpy as np

# %% Model and SAE setup
import os
os.environ['TORCH_USE_CUDA_DSA']='1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("Initializing model and SAE...")
model_state = setup_model_and_sae()

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# %% Clear experiment files
from models import ExperimentResults 

clear = True
if clear:
    ExperimentResults.clear_all_experiment_files()

# %% Prompt and configuration
prompt = \
""""
"You are an AI that solves problems using detailed reasoning. 
You should try to evaluate your thought process as you go along, 
and revise it if there are inconsistencies using phrases like (so, I said; wait, let me think; etc).
This is NOT a multiple choice question. If numbers are not provided, they are not needed.
Question: 
Answer:
"""

# Configuration based on store_mode
store_mode = "off"
max_new_tokens = 250
num_runs = 3

config = GenerationConfig(
    store_mode=store_mode,
    num_runs=1 if store_mode in ["memory"] else num_runs,
    # max_new_tokens=100 if store_mode in ["off", "memory"] else max_new_tokens,
    save_every_n_steps=1 if store_mode == "disk" else None
)

results = run_multiple_experiments(
    prompt=prompt,
    num_runs=config.num_runs,
    model_state=model_state,
    config=config
)

# Access internals (if applicable)
if config.store_mode == "memory":
    # Access all internals directly
    run0_internals = results.generation_internals[0]  # First run
    step0_internals = run0_internals[0]  # First step
elif config.store_mode == "disk":
    # Load from disk as needed
    numpy_data = np.load(f"{results.experiment_path}/runs/run_000/step_0000.npz")

# Visualize results
visualize_experiment_results(results)

# Clean up
torch.cuda.empty_cache()
gc.collect()

# %% [markdown]
# # Results outputs
# - model_state: The model state used for the experiment
# - config: The generation configuration used
# - prompt: The input prompt used
# - all_texts: List of all generated texts
# - stopping_reasons: Counter of early stopping reasons
# - token_frequencies: Counter of most common tokens
# - avg_length: Average length of generations
# - unique_ratio: Average ratio of unique tokens
# - generation_acts: Optional list of activation patterns
# - metadata: Additional experiment metadata
# 
# 
# It seems like we want the following:
# - All_texts
# - Generation_acts
# 
# And I think there is some stuff missing... most of the outputs are not that useful.


#%%

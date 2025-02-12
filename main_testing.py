# %%
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

# %%
import os
os.environ['TORCH_USE_CUDA_DSA']='1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("Initializing model and SAE...")
model_state = setup_model_and_sae()

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# %%
from models import ExperimentResults 

clear = False
if clear:
    ExperimentResults.clear_all_experiment_files()

# %%
prompt = \
""""
"You are an AI that solves problems step-by-step with detailed reasoning. 
Always explain your thought process using phrases like (so, I said; wait, let me think; etc).
Some other minor information: This is NOT a multiple choice question. If numbers are not provided, they are not needed. Good luck!
Question: If a ball is thrown by a person at a wall that is relatively close, what will happen to the ball shortly after the throw? 
Answer:
"""

config = GenerationConfig()
# Run experiments
with tqdm(total=config.num_runs, desc="Generating responses") as pbar:
    results = run_multiple_experiments(
        prompt=prompt,
        num_runs=config.num_runs,  
        config=config,
        model_state=model_state,
        progress_callback=lambda: pbar.update(1)
    )

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

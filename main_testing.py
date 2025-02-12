# %%
%load_ext autoreload
%autoreload 2

from config import GenerationConfig
from setup import setup_model_and_sae
from experiment import run_multiple_experiments
from visualization import visualize_experiment_results
import torch
import gc
from tqdm import tqdm

# %%
os.environ['TORCH_USE_CUDA_DSA']='1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("Initializing model and SAE...")
model_state = setup_model_and_sae()

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# %%
prompt = "You are an AI that solves problems step-by-step with detailed reasoning. Break the problem down into chunks that you can solve. \
Verify the correctness of these chunks before moving on. Always explain your thought process before providing the final answer. \
Solve your problem in the present tense. This is NOT a multiple choice question. If numbers are not provided, they are not needed. \
Question: If a ball is thrown by a person at a wall that is 10 feet away, what will happen to the ball within 5 seconds of the throw? \
Answer:"

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



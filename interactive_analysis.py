#%%
"""
Interactive Analysis Script

This script provides an interactive interface for analyzing SAE behavior
during text generation using Jupyter-style cells in VS Code.
"""

from .config import GenerationConfig
from .setup import setup_model_and_sae
from .experiment import run_generation_experiment
import torch

#%% Model initialization cell (run once)
print("Initializing model, tokenizer, and SAE...")
model, tokenizer, sae = setup_model_and_sae()
print("Initialization complete!")

#%% Analysis cell (run this for each new prompt)
# Clear GPU memory
torch.cuda.empty_cache()

# Clean up previous variables if they exist
for var in ['gen_acts', 'gen_texts', 'tokens']:
    if var in locals() or var in globals():
        exec(f'del {var}')

# Set your prompt
prompt = "Answer the following question: Q: What will happen if a ball is thrown at a wall? A:"

# Use creative configuration for more diverse outputs
config = GenerationConfig.precise()

print("Running generation with creative configuration...")
run_generation_experiment(
    prompt=prompt,
    config=config,
    model=model,
    tokenizer=tokenizer,
    sae=sae
)

#%% Try another prompt with precise configuration
prompt = "Write a short story about a magical forest: "

# Use precise configuration for more focused outputs
config = GenerationConfig.precise()

print("Running generation with precise configuration...")
run_generation_experiment(
    prompt=prompt,
    config=config,
    model=model,
    tokenizer=tokenizer,
    sae=sae
) 
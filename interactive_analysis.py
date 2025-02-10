#%%
from sae_investigations import (
    GenerationConfig,
    setup_model_and_sae,
    run_generation_experiment
)

import torch

#%% Model initialization cell (run once)
# Set up model, tokenizer, and SAE
model, tokenizer, sae = setup_model_and_sae()

#%% Analysis cell (run this for each new prompt)
# Clear GPU memory
torch.cuda.empty_cache()

# Set your prompt
prompt = "Answer the following question: Q: What will happen if a ball is thrown at a wall? A:"

# Use creative configuration
config = GenerationConfig.creative()

# Run experiment
run_generation_experiment(
    prompt=prompt,
    config=config,
    model=model,
    tokenizer=tokenizer,
    sae=sae
)

#%% Try another prompt
prompt = "Write a short story about a magical forest: "
run_generation_experiment(
    prompt=prompt,
    config=config,
    model=model,
    tokenizer=tokenizer,
    sae=sae
) 
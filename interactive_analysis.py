#%% Model initialization cell (run ONCE only)
import torch
print("Initializing model, tokenizer, and SAE...")
model_state = setup_model_and_sae()
print("Initialization complete!")

#%% Analysis cell (run this for testing, includes auto-reload)
import importlib
import experiment
import config
import setup
import visualization

# Reload modules to get latest changes
importlib.reload(experiment)
importlib.reload(config)
importlib.reload(setup)
importlib.reload(visualization)

# Re-import after reload
from config import GenerationConfig
from setup import setup_model_and_sae
from experiment import run_generation_experiment, run_multiple_experiments
from visualization import visualize_generation_activations, visualize_experiment_results

# Clear GPU memory
torch.cuda.empty_cache()

# Clean up previous variables if they exist
for var in ['gen_acts', 'gen_texts', 'tokens', 'results', 'fig']:
    if var in locals() or var in globals():
        exec(f'del {var}')

# Set your prompt
prompt = "Answer the following question: \
Q: What will happen if a ball is thrown at a wall very fast? \
A:"

# Use precise configuration for consistent outputs
config = GenerationConfig.precise()

# Run multiple experiments
results = run_multiple_experiments(
    prompt=prompt,
    num_runs=10,  # Adjust based on how many runs you want
    config=config,
    model_state=model_state
)

# Create and display visualization
fig = visualize_experiment_results(results)
fig.show()

# Print detailed statistics
print("\nDetailed Statistics:")
print(f"Average generation length: {results.avg_length:.2f} words")
print(f"Unique token ratio: {results.unique_ratio:.2%}")
print("\nStopping reasons:")
for reason, count in results.stopping_reasons.most_common():
    print(f"- {reason}: {count} times")
print("\nMost common tokens:")
for token, count in results.token_frequencies.most_common(10):
    print(f"- '{token}': {count} times")

# %%

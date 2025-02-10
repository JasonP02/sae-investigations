#%% Model initialization cell (run ONCE only)
import torch
from setup import setup_model_and_sae
print("Initializing model, tokenizer, and SAE...")
model_state = setup_model_and_sae()
print("Initialization complete!")

#%% Analysis cell (run this for testing, includes auto-reload)
import importlib
import experiment
import config
import setup
import visualization
import models

# Reload modules to get latest changes
importlib.reload(models)
importlib.reload(experiment)
importlib.reload(config)
importlib.reload(setup)
importlib.reload(visualization)

# Re-import after reload
from config import GenerationConfig
from setup import setup_model_and_sae
from experiment import run_multiple_experiments
from visualization import visualize_generation_activations, visualize_experiment_results
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter/IPython environments

# Memory management
import gc
import torch

# Clear CUDA cache and run garbage collection
torch.cuda.empty_cache()
gc.collect()

# Clean up previous variables if they exist
for var in ['gen_acts', 'gen_texts', 'tokens', 'results', 'fig']:
    if var in locals() or var in globals():
        exec(f'del {var}')

# Set your prompt
prompt = "Answer the following question and provide your reasoning for the answer: \
Q: What will happen if a ball is thrown at a wall very fast? \
A:"

# Use precise configuration for consistent outputs
config = GenerationConfig.precise()

# Run multiple experiments, returns ExperimentResults object
print(f"\nRunning {config.num_runs} experiments with prompt:\n{prompt}\n")

with tqdm(total=config.num_runs, desc="Generating responses") as pbar:
    results = run_multiple_experiments(
        prompt=prompt,
        num_runs=config.num_runs,  
        config=config,
        model_state=model_state,
        progress_callback=lambda: pbar.update(1)
    )

'''
results structure (ExperimentResults object):
    Attributes:
        model_state: The model state used for the experiment
        config: The generation configuration used
        prompt: The input prompt used
        all_texts: List of all generated texts
        stopping_reasons: Counter of early stopping reasons
        token_frequencies: Counter of most common tokens
        avg_length: Average length of generations
        unique_ratio: Average ratio of unique tokens
        generation_acts: Optional list of activation patterns
        metadata: Additional experiment metadata
    """
'''

#%% Visualize activations for first generation
print("\nDisplaying activation visualizations (original MLP vs SAE-encoded)...")
activation_figures = visualize_generation_activations(
    results.generation_acts[0],  # First run's activations
    results.all_texts[0],  # First run's texts
    title_prefix=f"{model_state.model_name} - {model_state.sae_name}"
)

for fig in activation_figures:
    try:
        fig.show()
        print("\nFigure shows:")
        if 'Feature Activations' in fig.layout.title.text:
            print("- Top: Original MLP layer activations (dense)")
            print("- Bottom: SAE-encoded activations (sparse)")
            print("- X-axis shows the full hidden dimension (top) vs learned SAE features (bottom)")
            print("- Y-axis shows generation steps with the last 20 chars of text at each step")
        else:
            print("- Evolution of top SAE features over generation steps")
            print("- Shows which features are most active and when they activate")
            print("- Hover over points to see the token generated at each step")
    except Exception as e:
        print(f"Error displaying figure: {e}")

#%% Visualize experiment results

# Create and display visualizations
figures = visualize_experiment_results(results, config.num_runs)
for fig in figures:
    try:
        fig.show()
    except Exception as e:
        print(f"Error displaying figure: {e}")

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


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

# Clear GPU memory
torch.cuda.empty_cache()

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



# Use tqdm for production runs
with tqdm(total=config.num_runs, desc="Generating responses") as pbar:
    results = run_multiple_experiments(
        prompt=prompt,
        num_runs=config.num_runs,  
        config=config,
        model_state=model_state,
        progress_callback=lambda: pbar.update(1)
    )

#%% Visualize activations for first generation

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

# Use precise configuration for consistent outputs
config = GenerationConfig.precise()

print("\nVisualizing activations for first generation...")
# Get the first generation's activations and text
first_gen_acts = results.generation_acts[0]  # Get first run's activations
first_gen_texts = []

# Create list of texts at each step by tracking the incremental changes
current_text = prompt
for i in range(len(first_gen_acts)):
    if i == 0:
        first_gen_texts.append(current_text)  # Start with prompt
    else:
        # Get the new text generated at this step
        new_text = results.all_texts[0]
        # Only add text if it's different from the last one
        if len(first_gen_texts) == 0 or new_text != first_gen_texts[-1]:
            first_gen_texts.append(new_text)

# Create activation visualizer with both original and encoded activations
activation_data = {
    'generation_acts': first_gen_acts,
    'generated_texts': first_gen_texts,
    'metadata': {
        **results.metadata,
        'model_name': results.model_state.model_name,
        'sae_name': results.model_state.sae_name
    }
}

# Create and display activation visualizations
print("\nDisplaying activation visualizations (original MLP vs SAE-encoded)...")
activation_viz = visualization.GenerationActivationVisualizer(activation_data)
activation_figures = activation_viz.create_figures()

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
        print(f"Error displaying activation figure: {e}")

#%% Visualize experiment results

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

# Use precise configuration for consistent outputs
config = GenerationConfig.precise()

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

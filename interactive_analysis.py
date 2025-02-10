#%% Model initialization cell (run ONCE only)
import torch
from setup import setup_model_and_sae
print("Initializing model, tokenizer, and SAE...")
model, tokenizer, sae = setup_model_and_sae()
print("Initialization complete!")

#%% Analysis cell (run this for testing, includes auto-reload)
import importlib
import experiment
import config
import setup
import visualization
import reasoning_analysis

# Reload modules to get latest changes
importlib.reload(experiment)
importlib.reload(config)
importlib.reload(setup)
importlib.reload(visualization)
importlib.reload(reasoning_analysis)

# Re-import after reload
from config import GenerationConfig
from setup import setup_model_and_sae
from experiment import run_generation_experiment, run_multiple_experiments
from visualization import visualize_generation_activations, visualize_experiment_results, visualize_reasoning_analysis
from reasoning_analysis import evaluate_prompt_effectiveness, suggest_prompt_improvements

# Clear GPU memory
torch.cuda.empty_cache()

# Clean up previous variables if they exist
for var in ['gen_acts', 'gen_texts', 'tokens', 'results', 'fig', 'analysis_results']:
    if var in locals() or var in globals():
        exec(f'del {var}')

# Set your base question
base_question = "What will happen if a ball is thrown at a wall very fast?"

# Use reasoning-focused configuration
config = GenerationConfig.reasoning_focused()

# Evaluate different prompt variants
analysis_results = evaluate_prompt_effectiveness(
    base_question=base_question,
    config=config,
    num_variants=5,  # Test 5 different prompt variants
    runs_per_variant=2  # Run each variant twice
)

# Print analysis results
print("\nPrompt Effectiveness Analysis:")
print("\nScores for each prompt variant:")
for prompt, score in analysis_results['prompt_scores'].items():
    print(f"\nPrompt: {prompt}")
    print(f"Score: {score:.2f}")

print("\nBest performing prompt:")
print(analysis_results['best_prompt'])

print("\nSuggested improvements:")
for suggestion in suggest_prompt_improvements(analysis_results):
    print(f"- {suggestion}")

#%% Visualization cell (optional, run to see plots)
# Create and display the reasoning analysis visualization
fig = visualization.visualize_reasoning_analysis(analysis_results)
fig.show()

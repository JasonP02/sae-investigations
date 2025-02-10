#%% Model initialization cell (run ONCE only)
import torch
print("Initializing model, tokenizer, and SAE...")
model, tokenizer, sae = setup_model_and_sae()
print("Initialization complete!")

#%% Analysis cell (run this for testing, includes auto-reload)
import importlib
import experiment
import config
import setup
import visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Reload modules to get latest changes
importlib.reload(experiment)
importlib.reload(config)
importlib.reload(setup)
importlib.reload(visualization)

# Re-import after reload
from config import GenerationConfig
from setup import setup_model_and_sae
from experiment import run_generation_experiment, run_multiple_experiments
from visualization import visualize_generation_activations

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
    model=model,
    tokenizer=tokenizer,
    sae=sae,
)

# Create visualization subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Early Stopping Reasons',
        'Most Common Tokens',
        'Generation Statistics',
        'Sample Generations'
    )
)

# 1. Early stopping reasons pie chart
labels = list(results['stopping_reasons'].keys())
values = list(results['stopping_reasons'].values())
fig.add_trace(
    go.Pie(labels=labels, values=values, textinfo='label+percent'),
    row=1, col=1
)

# 2. Most common tokens bar chart
tokens = list(results['token_frequencies'].keys())[:10]  # Top 10 tokens
counts = list(results['token_frequencies'].values())[:10]
fig.add_trace(
    go.Bar(x=tokens, y=counts, name='Token Frequency'),
    row=1, col=2
)

# 3. Statistics
fig.add_trace(
    go.Indicator(
        mode="number+delta",
        value=results['avg_length'],
        title="Avg Length",
        domain={'row': 1, 'column': 1}
    ),
    row=2, col=1
)

fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=results['unique_ratio'] * 100,
        title="Unique Token Ratio %",
        gauge={'axis': {'range': [0, 100]}},
        domain={'row': 1, 'column': 1}
    ),
    row=2, col=1
)

# 4. Sample generations table
sample_texts = results['all_texts'][:5]  # Show first 5 generations
fig.add_trace(
    go.Table(
        header=dict(values=['Sample Generations']),
        cells=dict(values=[sample_texts])
    ),
    row=2, col=2
)

# Update layout
fig.update_layout(
    height=1000,
    width=1200,
    showlegend=False,
    title_text="Generation Analysis Results"
)

fig.show()

# Print detailed statistics
print("\nDetailed Statistics:")
print(f"Average generation length: {results['avg_length']:.2f} words")
print(f"Unique token ratio: {results['unique_ratio']:.2%}")
print("\nStopping reasons:")
for reason, count in results['stopping_reasons'].most_common():
    print(f"- {reason}: {count} times")
print("\nMost common tokens:")
for token, count in results['token_frequencies'].most_common(10):
    print(f"- '{token}': {count} times")

# %%

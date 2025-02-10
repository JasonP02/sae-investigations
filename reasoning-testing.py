#%%
from sparsify import Sae

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import plotly.express as px
import numpy as np

import plotly.graph_objects as go
import plotly.subplots as sp

# What is the structure of these transcoders?
## First, lets look at the SAE attributes: 
# cfg returns: SaeConfig(expansion_factor=64, normalize_decoder=True, num_latents=65536, k=32, multi_topk=False, skip_connection=False)

# the other parameters are: 
# - din 1536,  
# - latents: 65536, 
# - encoder: Linear(in_features=1536, out_features=65536, bias=True), 
# - decoder: torch.Size([65536, 1536]), 
# - decoder_bias: torch.Size([1536]), 
# - skip: None
# - dtype: torch.float32

## Output information
# Latent acts contains two tensors of equal shape: (top acts, top indices)

# - (1,9,32) for each layer (there are 15 layers)
# - 1 is: batch size
# - 9 is: sequence length
# - 32 is: k (number of top features)


## How can we use this information?
# First, I want to look at the distribution of the latent activations in a 2x2 plot (sequence length, k)
#%%
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize SAE and tokenizer
sae = Sae.load_from_hub(
    "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k", 
    hookpoint="layers.10.mlp",
    device=device
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Load model and move to GPU if available
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device_map={"": device},
    torch_dtype=torch.float32
).to(device)
#%%

def analyze_generation(
    model,
    tokenizer,
    sae,
    input_text: str,
    max_new_tokens: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    pad_token_id: int = None,
    eos_token_id: int = None,
):
    """
    Analyze the SAE activations during the generation process.
    """
    inputs = tokenizer(input_text, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    
    # Store results
    generation_acts = []
    generated_texts = []
    
    with torch.inference_mode():
        # Move to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Initial forward pass to get hidden states for input
        outputs = model(**inputs, output_hidden_states=True)
        current_ids = inputs["input_ids"]
        
        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Get next token probabilities
            next_token_logits = outputs.logits[:, -1, :]
            
            if do_sample:
                # Temperature scaling
                next_token_logits = next_token_logits / temperature
                # Top-p sampling
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_indices[0, torch.multinomial(sorted_probs[0], 1)]
            else:
                # Greedy decoding
                next_token = next_token_logits.argmax(dim=-1)
            
            # Add new token to sequence
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Forward pass with new sequence
            outputs = model(current_ids, output_hidden_states=True)
            
            # Collect activations for this step
            step_acts = []
            for hidden_state in outputs.hidden_states:
                step_acts.append(sae.encode(hidden_state))
            generation_acts.append(step_acts)
            
            # Store generated text
            generated_texts.append(tokenizer.decode(current_ids[0], skip_special_tokens=True))
            
            # Optional: print progress
            print(f"Step {len(generation_acts)}/{max_new_tokens}: {generated_texts[-1]}")
            
            # Check for EOS token
            if next_token.item() == (eos_token_id or tokenizer.eos_token_id):
                break
    
    return generation_acts, generated_texts, current_ids[0]

def visualize_generation_activations(
    generation_acts,
    generated_texts,
    layer_idx=10,
    step_interval=1  # Show every nth step
):
    """
    Visualize how activations change during the generation process.
    
    Args:
        generation_acts: List of activations for each generation step
        generated_texts: List of generated text at each step
        layer_idx: Which transformer layer to analyze
        step_interval: Show every nth step to reduce visual complexity
    """
    figures = []
    
    # 1. Feature activation heatmap over generation steps
    fig_gen_heatmap = go.Figure()
    
    # Get activations for specified layer across all steps
    step_acts = []
    step_indices = []
    step_labels = []
    
    for step, (acts, text) in enumerate(zip(generation_acts, generated_texts)):
        if step % step_interval == 0:
            layer_acts = acts[layer_idx]
            step_acts.append(layer_acts.top_acts.detach().cpu().numpy()[0, -1])  # Last token
            step_indices.append(layer_acts.top_indices.detach().cpu().numpy()[0, -1])
            step_labels.append(f"Step {step}: {text[-20:]}")  # Show last 20 chars
    
    # Convert to numpy arrays
    step_acts = np.array(step_acts)
    step_indices = np.array(step_indices)
    
    # Activation values heatmap
    fig_gen_heatmap.add_trace(
        go.Heatmap(
            z=step_acts,
            y=step_labels,
            colorscale='RdBu',
            name='Activation Values',
            showscale=True,
            colorbar=dict(title="Activation Value")
        )
    )
    
    fig_gen_heatmap.update_layout(
        title_text=f"Feature Activations During Generation (Layer {layer_idx})",
        xaxis_title="K-index",
        yaxis_title="Generation Step",
        height=800,
        width=1200
    )
    figures.append(fig_gen_heatmap)
    
    # 2. Feature usage evolution
    fig_feature_evolution = go.Figure()
    
    # Track top N most used features
    N = 10
    feature_counts = np.zeros(65536)  # Total number of latents
    for step_idx in step_indices:
        unique, counts = np.unique(step_idx, return_counts=True)
        feature_counts[unique] += counts
    
    top_features = np.argsort(feature_counts)[-N:]
    
    # Plot activation strength of top features over time
    for feature_idx in top_features:
        feature_activations = []
        for step_act, step_idx in zip(step_acts, step_indices):
            # Find if feature was used in this step
            feature_pos = np.where(step_idx == feature_idx)[0]
            if len(feature_pos) > 0:
                feature_activations.append(step_act[feature_pos[0]])
            else:
                feature_activations.append(0)
        
        fig_feature_evolution.add_trace(
            go.Scatter(
                y=feature_activations,
                name=f"Feature {feature_idx}",
                mode='lines+markers'
            )
        )
    
    fig_feature_evolution.update_layout(
        title_text=f"Top {N} Feature Evolution During Generation (Layer {layer_idx})",
        xaxis_title="Generation Step",
        yaxis_title="Activation Strength",
        height=600,
        width=1000,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        )
    )
    figures.append(fig_feature_evolution)
    
    return figures

#%% Initialize models (run this once)


#%% Analysis cell (run this for each new prompt)
# Clear previous outputs
if 'gen_acts' in locals() or 'gen_acts' in globals():
    del gen_acts
if 'gen_texts' in locals() or 'gen_texts' in globals():
    del gen_texts
if 'tokens' in locals() or 'tokens' in globals():
    del tokens
torch.cuda.empty_cache()

# Set your prompt
in_text = "Q: What is the value of x for the equation x = 2 + 1? A: x = 3 Q: What is the value of y for the equation y = 2 + 1? A: y = "

# Get generation analysis
gen_acts, gen_texts, tokens = analyze_generation(
    model=model,
    tokenizer=tokenizer,
    sae=sae,
    input_text=in_text,
    max_new_tokens=10,
    temperature=0.7,
    top_p=0.9,
)

# Print the final generated text
print("\nFinal generated text:", gen_texts[-1])

#%% Visualization cell (run this to see the results)
# Visualize generation process
gen_figs = visualize_generation_activations(gen_acts, gen_texts)
for fig in gen_figs:
    fig.show()



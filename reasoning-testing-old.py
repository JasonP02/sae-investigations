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
    torch_dtype=torch.float16,  # Use half precision for 6GB VRAM
    # Add memory and performance optimizations
    low_cpu_mem_usage=True,
    max_memory={0: "5GB"},  # Reserve some VRAM for other operations
    use_flash_attention_2=False,  # RTX 2060 doesn't support it
).to(device)

try:
    print("Attempting to compile model for speed...")
    # Use conservative compilation settings for RTX 2060
    model = torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=False,
        options={
            "max_autotune": False,
            "max_parallel_gemm": 1,  # More conservative for 6GB VRAM
            "triton.cudagraphs": False,  # Disable cudagraphs for older GPUs
        }
    )
    print("Model compiled successfully!")
except Exception as e:
    print(f"Could not compile model (requires PyTorch 2.0+): {e}")
    print("Continuing without compilation...")

# Also move SAE to half precision for consistency
sae = sae.to(torch.float16)
#%%

# Configuration class for generation analysis
class GenerationConfig:
    """
    Configuration class for text generation analysis.
    
    Attributes:
        Generation Parameters:
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness in generation (higher = more random)
            top_p (float): Nucleus sampling parameter (higher = more diverse)
            do_sample (bool): Whether to use sampling vs greedy decoding
            min_confidence (float): Minimum confidence threshold for token generation
        
        Repetition Control:
            repetition_window (int): Number of recent tokens to check for repetition
            max_repetition_ratio (float): Maximum allowed ratio of repeated tokens
            ngram_size (int): Size of ngrams to check for repetition
            max_ngram_repeats (int): Maximum times an ngram can repeat
            min_unique_ratio (float): Minimum ratio of unique tokens required
            semantic_similarity_threshold (float): Threshold for semantic similarity between phrases
        
        Special Tokens:
            pad_token_id (Optional[int]): ID of padding token
            eos_token_id (Optional[int]): ID of end-of-sequence token
    """
    def __init__(
        self,
        # Generation parameters
        max_new_tokens: int = 100,
        temperature: float = 0.9,
        top_p: float = 0.92,
        do_sample: bool = True,
        min_confidence: float = 0.1,
        
        # Repetition control
        repetition_window: int = 12,
        max_repetition_ratio: float = 0.5,
        ngram_size: int = 6,
        max_ngram_repeats: int = 2,
        min_unique_ratio: float = 0.4,
        semantic_similarity_threshold: float = 0.8,
        
        # Special tokens
        pad_token_id: int = None,
        eos_token_id: int = None,
    ):
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.min_confidence = min_confidence
        
        # Repetition control
        self.repetition_window = repetition_window
        self.max_repetition_ratio = max_repetition_ratio
        self.ngram_size = ngram_size
        self.max_ngram_repeats = max_ngram_repeats
        self.min_unique_ratio = min_unique_ratio
        self.semantic_similarity_threshold = semantic_similarity_threshold
        
        # Special tokens
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
    
    @classmethod
    def default(cls) -> 'GenerationConfig':
        """Creates a default configuration optimized for general use."""
        return cls()
    
    @classmethod
    def creative(cls) -> 'GenerationConfig':
        """Creates a configuration optimized for creative, diverse outputs."""
        return cls(
            temperature=1.0,
            top_p=0.95,
            min_confidence=0.05,
            repetition_window=8,
            max_ngram_repeats=3,
            min_unique_ratio=0.3
        )
    
    @classmethod
    def precise(cls) -> 'GenerationConfig':
        """Creates a configuration optimized for precise, focused outputs."""
        return cls(
            temperature=0.7,
            top_p=0.85,
            min_confidence=0.2,
            repetition_window=16,
            max_ngram_repeats=1,
            min_unique_ratio=0.5
        )

def analyze_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: Sae,
    input_text: str,
    config: GenerationConfig = None,
) -> tuple[list, list, torch.Tensor]:
    """
    Analyzes the generation process of a language model with sparse autoencoder integration.
    
    This function performs token generation while monitoring various aspects such as:
    - Token confidence and probabilities
    - Repetition patterns (local and n-gram based)
    - Semantic similarity between generated phrases
    - Feature activations in the specified layer
    
    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer corresponding to the model
        sae: Sparse autoencoder for analyzing internal representations
        input_text: The prompt text to start generation from
        config: Generation configuration parameters (uses default if None)
    
    Returns:
        tuple containing:
        - generation_acts: List of activation patterns for each generation step
        - generated_texts: List of generated text at each step
        - tokens: Final sequence of generated tokens
    
    Example:
        >>> config = GenerationConfig.creative()  # Use creative configuration
        >>> acts, texts, tokens = analyze_generation(model, tokenizer, sae, "Once upon a time", config)
    """
    if config is None:
        config = GenerationConfig.default()
    
    inputs = tokenizer(input_text, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    
    # Store results
    generation_acts = []
    generated_texts = []
    recent_phrases = []
    
    # Move repetition tracking to GPU
    recent_tokens = torch.zeros(config.repetition_window, dtype=torch.long, device=model.device)
    all_generated_tokens = []
    
    # Common filler patterns to detect
    filler_patterns = [
        "is a", "is an", "is the", "and the", "then the",
        "to the", "of the", "in the", "on the", "at the",
        "and then", "so the", "which is", "that is"
    ]
    max_consecutive_fillers = 2
    consecutive_fillers = 0
    
    with torch.inference_mode():
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        current_ids = inputs["input_ids"]
        
        # Initial forward pass
        outputs = model(current_ids, output_hidden_states=True)
        
        # Pre-allocate tensors for efficiency
        next_token = torch.zeros(1, 1, dtype=torch.long, device=model.device)
        
        # Track the last few complete phrases for semantic similarity
        last_complete_phrase = ""
        current_phrase = ""
        phrase_end_tokens = {'.', '!', '?', ',', ';', ':'}
        
        # Generate tokens one at a time
        for step in range(config.max_new_tokens):
            # Get next token probabilities
            next_token_logits = outputs.logits[:, -1:, :]
            
            if config.do_sample:
                next_token_logits = next_token_logits / config.temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Check confidence
                top_prob = probs.max().item()
                if top_prob < config.min_confidence:
                    print(f"Stopping: Low confidence ({top_prob:.3f})")
                    break
                
                # Nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs > config.top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                
                sample_idx = torch.multinomial(sorted_probs[0, 0], 1)
                next_token[0, 0] = sorted_indices[0, 0, sample_idx]
            else:
                next_token[0, 0] = next_token_logits[0, 0].argmax()
            
            # Track token for repetition checks
            token_val = next_token[0, 0].item()
            token_text = tokenizer.decode([token_val])
            all_generated_tokens.append(token_val)
            recent_tokens[step % config.repetition_window] = token_val
            
            # Update current phrase
            current_phrase += token_text
            if any(token_text.endswith(end_token) for end_token in phrase_end_tokens):
                if len(current_phrase.strip()) > 0:
                    recent_phrases.append(current_phrase.strip())
                    if len(recent_phrases) > 3:  # Keep last 3 phrases
                        recent_phrases.pop(0)
                current_phrase = ""
            
            # Check for semantic repetition in recent phrases
            if len(recent_phrases) >= 2:
                last_two_phrases = recent_phrases[-2:]
                # Simple similarity check based on common words
                words1 = set(last_two_phrases[0].lower().split())
                words2 = set(last_two_phrases[1].lower().split())
                if len(words1) > 0 and len(words2) > 0:
                    similarity = len(words1.intersection(words2)) / min(len(words1), len(words2))
                    if similarity > config.semantic_similarity_threshold:
                        print(f"Stopping: High semantic similarity ({similarity:.2f})")
                        break
            
            # Check for filler patterns
            current_text = tokenizer.decode(current_ids[0])
            if any(pattern in current_text[-20:].lower() for pattern in filler_patterns):
                consecutive_fillers += 1
                if consecutive_fillers > max_consecutive_fillers:
                    print(f"Stopping: Too many consecutive filler phrases")
                    break
            else:
                consecutive_fillers = 0
            
            # Check for local repetition (in recent window)
            if step >= config.repetition_window:
                unique_tokens = torch.unique(recent_tokens)
                unique_ratio = len(unique_tokens) / config.repetition_window
                if unique_ratio < config.min_unique_ratio:
                    print(f"Stopping: Low unique token ratio ({unique_ratio:.2f})")
                    break
            
            # Check for ngram repetition
            if len(all_generated_tokens) >= config.ngram_size:
                # Get the last ngram
                current_ngram = tuple(all_generated_tokens[-config.ngram_size:])
                # Count occurrences in previous tokens
                ngram_count = 0
                for i in range(len(all_generated_tokens) - config.ngram_size):
                    if tuple(all_generated_tokens[i:i+config.ngram_size]) == current_ngram:
                        ngram_count += 1
                if ngram_count >= config.max_ngram_repeats:
                    print(f"Stopping: Ngram repeated {ngram_count} times")
                    break
            
            # Concatenate and forward pass
            current_ids = torch.cat([current_ids, next_token], dim=1)
            outputs = model(current_ids, output_hidden_states=True)
            
            # Collect activations
            step_acts = []
            for hidden_state in [outputs.hidden_states[10]]:
                step_acts.append(sae.encode(hidden_state))
            generation_acts.append(step_acts)
            
            # Store generated text
            if step % 5 == 0 or step == config.max_new_tokens - 1:
                generated_texts.append(tokenizer.decode(current_ids[0], skip_special_tokens=True))
                if step % 5 == 0:
                    print(f"Step {step}: Confidence = {top_prob:.3f}")
            
            # Check for EOS token
            if token_val == (config.eos_token_id or tokenizer.eos_token_id):
                print("Stopping: EOS token generated")
                break
    
    # Fill in missing texts
    while len(generated_texts) < len(generation_acts):
        generated_texts.append(tokenizer.decode(current_ids[0], skip_special_tokens=True))
    
    return generation_acts, generated_texts, current_ids[0]

def visualize_generation_activations(
    generation_acts,
    generated_texts,
    layer_idx=0,  # Changed from 10 to 0 since we only collect one layer
    step_interval=1  # Show every nth step
):
    """
    Visualize how activations change during the generation process.
    
    Args:
        generation_acts: List of activations for each generation step
        generated_texts: List of generated text at each step
        layer_idx: Index within each step's activations (should be 0 since we only collect one layer)
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
            # Since we only collect one layer, it's the first item in acts
            layer_acts = acts[0]  # Changed from acts[layer_idx]
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
        title_text=f"Feature Activations During Generation (MLP Layer 10)",  # Updated title
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
    
    # Get the generated tokens for hover text
    all_tokens = []
    for text_idx in range(len(generated_texts)-1):
        if text_idx + 1 < len(generated_texts):
            # Get the new token by comparing with previous text
            new_token = generated_texts[text_idx + 1][len(generated_texts[text_idx]):]
            all_tokens.append(new_token if new_token else "")
    
    # Plot activation strength of top features over time
    for feature_idx in top_features:
        feature_activations = []
        hover_texts = []
        
        for step, (step_act, step_idx) in enumerate(zip(step_acts, step_indices)):
            # Find if feature was used in this step
            feature_pos = np.where(step_idx == feature_idx)[0]
            activation = step_act[feature_pos[0]] if len(feature_pos) > 0 else 0
            feature_activations.append(activation)
            
            # Create hover text with token info
            token_text = all_tokens[step-1] if step > 0 and step <= len(all_tokens) else ""
            hover_text = f"Step {step}<br>Feature: {feature_idx}<br>Activation: {activation:.3f}"
            if token_text:
                hover_text += f"<br>Token: '{token_text}'"
            hover_texts.append(hover_text)
        
        fig_feature_evolution.add_trace(
            go.Scatter(
                y=feature_activations,
                name=f"Feature {feature_idx}",
                mode='lines+markers',
                hovertext=hover_texts,
                hoverinfo='text'
            )
        )
    
    fig_feature_evolution.update_layout(
        title_text=f"Top {N} Feature Evolution During Generation (MLP Layer 10)",
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
        ),
        hovermode='closest'
    )
    figures.append(fig_feature_evolution)
    
    return figures

#%% Analysis cell (run this for each new prompt)
# Clear previous outputs and ensure clean GPU memory
if 'gen_acts' in locals() or 'gen_acts' in globals():
    del gen_acts
if 'gen_texts' in locals() or 'gen_texts' in globals():
    del gen_texts
if 'tokens' in locals() or 'tokens' in globals():
    del tokens
torch.cuda.empty_cache()

# Set your prompt
in_text = "Answer the following question: Q: What will happen if a ball is thrown at a wall? A:"

# Create generation configuration
gen_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.9,
    top_p=0.92,
    min_confidence=0.1,
    repetition_window=12,
    ngram_size=6,
    max_ngram_repeats=2,
    min_unique_ratio=0.4,
    semantic_similarity_threshold=0.8
)

# Get generation analysis
gen_acts, gen_texts, tokens = analyze_generation(
    model=model,
    tokenizer=tokenizer,
    sae=sae,
    input_text=in_text,
    config=gen_config
)

# Print generation progress
print("\nGeneration steps:", len(gen_texts))
print("Final text:", gen_texts[-1])

# Force garbage collection after generation
import gc
gc.collect()
torch.cuda.empty_cache()


#%% Visualization cell (run this to see the results)
# Visualize generation process
gen_figs = visualize_generation_activations(gen_acts, gen_texts)
for fig in gen_figs:
    fig.show()
# %%

from typing import List, Tuple, Set
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae # type: ignore
from config import GenerationConfig

# Common filler patterns to detect
FILLER_PATTERNS: List[str] = [
    "is a", "is an", "is the", "and the", "then the",
    "to the", "of the", "in the", "on the", "at the",
    "and then", "so the", "which is", "that is"
]

# Tokens that indicate phrase boundaries
PHRASE_END_TOKENS: Set[str] = {'.', '!', '?', ',', ';', ':'}

def analyze_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: Sae,
    input_text: str,
    config: GenerationConfig = None,
) -> Tuple[List, List, torch.Tensor]:
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
    
    consecutive_fillers = 0
    
    with torch.inference_mode():
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        current_ids = inputs["input_ids"]
        
        # Initial forward pass
        outputs = model(current_ids, output_hidden_states=True)
        
        # Pre-allocate tensors for efficiency
        next_token = torch.zeros(1, 1, dtype=torch.long, device=model.device)
        
        # Track phrases for semantic similarity
        current_phrase = ""
        
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
            if any(token_text.endswith(end_token) for end_token in config.phrase_end_tokens):
                if len(current_phrase.strip()) > 0:
                    recent_phrases.append(current_phrase.strip())
                    if len(recent_phrases) > config.max_recent_phrases:
                        recent_phrases.pop(0)
                current_phrase = ""
            
            # Check for semantic repetition in recent phrases
            if len(recent_phrases) >= 2:
                last_two_phrases = recent_phrases[-2:]
                words1 = set(last_two_phrases[0].lower().split())
                words2 = set(last_two_phrases[1].lower().split())
                if len(words1) > 0 and len(words2) > 0:
                    similarity = len(words1.intersection(words2)) / min(len(words1), len(words2))
                    if similarity > config.semantic_similarity_threshold:
                        print(f"Stopping: High semantic similarity ({similarity:.2f})")
                        break
            
            # Check for filler patterns
            current_text = tokenizer.decode(current_ids[0])
            if any(pattern in current_text[-config.phrase_context_window:].lower() for pattern in config.filler_patterns):
                consecutive_fillers += 1
                if consecutive_fillers > config.max_consecutive_fillers:
                    print(f"Stopping: Too many consecutive filler phrases")
                    break
            else:
                consecutive_fillers = 0
            
            # Check for local repetition
            if step >= config.repetition_window:
                unique_tokens = torch.unique(recent_tokens)
                unique_ratio = len(unique_tokens) / config.repetition_window
                if unique_ratio < config.min_unique_ratio:
                    print(f"Stopping: Low unique token ratio ({unique_ratio:.2f})")
                    break
            
            # Check for ngram repetition
            if len(all_generated_tokens) >= config.ngram_size:
                current_ngram = tuple(all_generated_tokens[-config.ngram_size:])
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
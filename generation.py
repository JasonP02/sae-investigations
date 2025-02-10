from typing import List, Tuple, Set, Optional, Dict, Any
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

def get_next_token(
    model: AutoModelForCausalLM,
    current_ids: torch.Tensor,
    config: GenerationConfig
) -> Tuple[torch.Tensor, float, Any]:
    """Handle the forward pass and token selection logic."""
    if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
        torch.compiler.cudagraph_mark_step_begin()
    outputs = model(current_ids, output_hidden_states=True)
    next_token_logits = outputs.logits[:, -1:, :]
    
    if config.do_sample:
        next_token_logits = next_token_logits / config.temperature
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        top_prob = probs.max().item()
        
        # Nucleus sampling
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum_probs > config.top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        sample_idx = torch.multinomial(sorted_probs[0, 0], 1)
        next_token = sorted_indices[0, 0, sample_idx].unsqueeze(0).unsqueeze(0)
    else:
        next_token = next_token_logits[0, 0].argmax().unsqueeze(0).unsqueeze(0)
        top_prob = 1.0
    
    return next_token, top_prob, outputs

def check_stopping_conditions(
    token_val: int,
    all_generated_tokens: List[int],
    config: GenerationConfig,
    top_prob: float,
    tokenizer: AutoTokenizer
) -> Tuple[bool, Optional[str]]:
    """Check all stopping conditions and return if should stop and why."""
    # Check confidence
    if top_prob < config.min_confidence:
        return True, f"Low confidence ({top_prob:.3f})"
    
    # Check for ngram repetition
    if len(all_generated_tokens) >= config.ngram_size:
        current_ngram = tuple(all_generated_tokens[-config.ngram_size:])
        ngram_count = 0
        for i in range(len(all_generated_tokens) - config.ngram_size):
            if tuple(all_generated_tokens[i:i+config.ngram_size]) == current_ngram:
                ngram_count += 1
        if ngram_count >= config.max_ngram_repeats:
            return True, f"Ngram repeated {ngram_count} times"
    
    # Check for EOS token
    if token_val == (config.eos_token_id or tokenizer.eos_token_id):
        return True, "EOS token generated"
    
    return False, None

def collect_activations(outputs, sae: Sae) -> List[Dict[str, torch.Tensor]]:
    """Extract and process activations from model outputs, keeping on GPU."""
    step_acts = []
    hidden_state = outputs.hidden_states[10].detach()  # Keep on GPU
    encoded = sae.encode(hidden_state)  # Keep on GPU
    
    step_acts.append({
        'original': hidden_state,
        'encoded': encoded
    })
    return step_acts

def analyze_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: Sae,
    input_text: str,
    config: GenerationConfig = None,
) -> Tuple[List, List, torch.Tensor, str]:
    """Main generation loop, now orchestrating the separate components."""
    current_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    all_generated_tokens = []
    generation_acts = []
    generated_texts = []
    stopping_reason = None

    try:
        for step in range(config.max_new_tokens):
            # Get next token
            with torch.no_grad():
                next_token, top_prob, outputs = get_next_token(model, current_ids, config)
            token_val = next_token[0, 0].item()
            all_generated_tokens.append(token_val)
            
            # Check stopping conditions
            should_stop, reason = check_stopping_conditions(
                token_val,
                all_generated_tokens,
                config,
                top_prob,
                tokenizer
            )
            if should_stop:
                stopping_reason = reason
                break
            
            # Collect activations and immediately clear outputs
            step_acts = collect_activations(outputs, sae)
            generation_acts.append(step_acts)
            del outputs, step_acts
            
            # Update state and clear memory
            current_ids = torch.cat([current_ids, next_token], dim=1)
            del next_token
            
            if step % 5 == 0 or step == config.max_new_tokens - 1:
                generated_texts.append(tokenizer.decode(current_ids[0], skip_special_tokens=True))
            
            # Aggressive memory clearing
            torch.cuda.empty_cache()
    finally:
        # Ensure memory is cleared even if an error occurs
        torch.cuda.empty_cache()
    
    return generation_acts, generated_texts, current_ids[0], stopping_reason
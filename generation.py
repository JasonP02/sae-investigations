from typing import List, Tuple, Set, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae # type: ignore
from config import GenerationConfig
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    with torch.no_grad():
        outputs = model(current_ids)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Apply temperature
        if config.temperature != 1.0:
            next_token_logits = next_token_logits / config.temperature
        
        # Get probabilities
        probs = torch.softmax(next_token_logits, dim=-1)
        
        if config.do_sample:
            # Apply top-p filtering
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                # Recompute probs after filtering
                probs = torch.softmax(next_token_logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy selection
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Get probability for selected token - ensure it's a scalar
        selected_prob = probs[0, next_token[0, 0]].detach().cpu().item()
    
    return next_token, selected_prob, outputs

def check_stopping_conditions(
    token_val: int,
    all_generated_tokens: List[int],
    config: GenerationConfig,
    top_prob: float,
    tokenizer: AutoTokenizer
) -> Tuple[bool, Optional[str]]:
    """Check all stopping conditions and return if should stop and why."""
    # top_prob should already be a scalar float at this point
    
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
    logger.debug("Starting analyze_generation")
    current_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    all_generated_tokens = []
    generation_acts = []
    generated_texts = []
    stopping_reason = None

    try:
        for step in range(config.max_new_tokens):
            print(f"\nStep {step} beginning")
            
            # Get next token
            print("Before get_next_token")
            with torch.no_grad():
                next_token, top_prob, outputs = get_next_token(model, current_ids, config)
            print(f"After get_next_token. top_prob: {top_prob}, type: {type(top_prob)}")
            
            token_val = next_token[0, 0].item()
            print(f"token_val: {token_val}")
            all_generated_tokens.append(token_val)
            
            # Ensure top_prob is a scalar float
            if torch.is_tensor(top_prob):
                top_prob = top_prob.item()
            
            # Check stopping conditions
            print(f"Before check_stopping_conditions. top_prob: {top_prob}, type: {type(top_prob)}")
            should_stop, reason = check_stopping_conditions(
                token_val,
                all_generated_tokens,
                config,
                float(top_prob),  # Explicitly cast to float
                tokenizer
            )
            print("After check_stopping_conditions")
            
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
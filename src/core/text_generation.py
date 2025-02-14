"""Text generation module with SAE analysis capabilities.

This module handles text generation with integrated SAE analysis, focusing on:
1. Token generation and sampling
2. Early stopping conditions
3. SAE activation tracking
"""

import logging
from typing import List, Tuple, Set, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from config import GenerationConfig
from models import ModelInternals, ExperimentResults
from einops import rearrange
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.propagate = False

FILLER_PATTERNS: List[str] = [
    "is a", "is an", "is the", "and the", "then the",
    "to the", "of the", "in the", "on the", "at the",
    "and then", "so the", "which is", "that is"
]

PHRASE_END_TOKENS: Set[str] = {'.', '!', '?', ',', ';', ':'}

class TokenGenerator:
    """Handles token generation and sampling logic."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: GenerationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.past_key_values = None  # For KV caching
    
    def get_next_token(self, current_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Generate next token using the model's predictions."""
        # Use KV cache if available
        input_ids = current_ids[:, -1:] if self.past_key_values is not None else current_ids
        
        outputs = self.model(
            input_ids,
            output_hidden_states=True,  # Keep getting all internals
            output_attentions=True,     # Keep getting all internals
            use_cache=True,             # Enable KV caching
            past_key_values=self.past_key_values
        )
        
        # Update KV cache
        self.past_key_values = outputs.past_key_values
        
        if torch.isnan(outputs.logits).any():
            raise RuntimeError("Model produced NaN logits")
            
        next_token_logits = outputs.logits[:, -1:, :].squeeze(1)
        
        # Temperature scaling
        if self.config.do_sample:
            next_token_logits = next_token_logits / self.config.temperature
            
        # Stable softmax
        next_token_logits = next_token_logits - next_token_logits.max(dim=-1, keepdim=True)[0]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        
        # Validate probabilities
        if torch.isnan(probs).any() or (probs < 0).any() or (probs > 1).any():
            raise RuntimeError("Invalid probability distribution")
            
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token, outputs, probs
    
    def reset_cache(self):
        """Reset the KV cache. Call this at the end of generation."""
        self.past_key_values = None

class StoppingCriteria:
    """Manages generation stopping conditions."""
    
    def __init__(self, config: GenerationConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.recent_tokens = torch.zeros(config.repetition_window, dtype=torch.long)
        self.consecutive_fillers = 0
        self.recent_phrases = []
        self.current_phrase = ""
    
    def should_stop(self, token: int, token_text: str, current_text: str, step: int) -> Tuple[bool, Optional[str]]:
        """Check if generation should stop based on various criteria."""
        # Update token history
        self.recent_tokens[step % self.config.repetition_window] = token
        self.current_phrase += token_text
        
        # Check EOS
        if token == (self.config.eos_token_id or self.tokenizer.eos_token_id):
            return True, "EOS token generated"
        
        # Check phrase similarity
        if any(token_text.endswith(end_token) for end_token in self.config.phrase_end_tokens):
            if self.current_phrase.strip():
                self.recent_phrases.append(self.current_phrase.strip())
                if len(self.recent_phrases) > self.config.max_recent_phrases:
                    self.recent_phrases.pop(0)
                    
                if len(self.recent_phrases) >= 2:
                    words1 = set(self.recent_phrases[-2].lower().split())
                    words2 = set(self.recent_phrases[-1].lower().split())
                    if words1 and words2:
                        similarity = len(words1.intersection(words2)) / min(len(words1), len(words2))
                        if similarity > self.config.semantic_similarity_threshold:
                            return True, f"High semantic similarity ({similarity:.2f})"
            self.current_phrase = ""
        
        # Check filler patterns
        if any(pattern in current_text[-self.config.phrase_context_window:].lower() 
               for pattern in self.config.filler_patterns):
            self.consecutive_fillers += 1
            if self.consecutive_fillers > self.config.max_consecutive_fillers:
                return True, "Too many consecutive filler phrases"
        else:
            self.consecutive_fillers = 0
            
        # Check diversity
        if step >= self.config.repetition_window:
            unique_ratio = len(torch.unique(self.recent_tokens)) / self.config.repetition_window
            if unique_ratio < self.config.min_unique_ratio:
                return True, f"Low unique token ratio ({unique_ratio:.2f})"
                
        return False, None

def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: Sae,
    input_text: str,
    results: ExperimentResults,
    run_idx: int,
    config: GenerationConfig = None,
) -> Tuple[List[ModelInternals], List[str], torch.Tensor, str]:
    """Generate text with integrated SAE analysis.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        sae: Sparse autoencoder
        input_text: Input prompt
        results: Results container
        run_idx: Current run index
        config: Generation configuration
        
    Returns:
        Tuple containing:
        - List of model internals
        - List of generated texts
        - Final token IDs
        - Stopping reason
    """
    if config is None:
        config = GenerationConfig()
        
    # Initialize components
    token_gen = TokenGenerator(model, tokenizer, config)
    stopping_criteria = StoppingCriteria(config, tokenizer)
    
    # Setup initial state
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    current_ids = inputs["input_ids"]
    
    generation_acts = []
    generated_texts = []
    generation_internals = []
    stopping_reason = None
    
    logger.info("Beginning generation loop")
    with torch.inference_mode():
        for step in range(config.max_new_tokens):
            try:
                # Generate next token
                next_token, outputs, probs = token_gen.get_next_token(current_ids)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                token_text = tokenizer.decode([next_token[0].item()])
                
                # Check stopping criteria
                current_text = tokenizer.decode(current_ids[0])
                should_stop, reason = stopping_criteria.should_stop(
                    next_token[0].item(), token_text, current_text, step
                )
                if should_stop:
                    stopping_reason = reason
                    break
                
                # Analyze SAE activations
                sae_input = outputs.hidden_states[10]
                encoded = sae.encode(sae_input)
                step_acts = [{
                    'original': outputs.hidden_states[10].detach(),
                    'encoded_acts': encoded[0],
                    'encoded_indices': encoded[1]
                }]
                generation_acts.append(step_acts)
                
                # Store results
                if step % 5 == 0 or step == config.max_new_tokens - 1:
                    current_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
                    generated_texts.append(current_text)
                
                # Store internals based on mode
                step_internals = ModelInternals(
                    hidden_states=outputs.hidden_states,
                    attention={i: outputs.attentions[i] for i in range(len(outputs.attentions))},
                    mlp={i: {
                        'pre_activation': outputs.hidden_states[i],
                        'post_activation': outputs.hidden_states[i+1]
                    } for i in range(len(outputs.hidden_states)-1)},
                    residual=outputs.hidden_states,
                    token=next_token[0].item(),
                    token_text=token_text,
                    logits=outputs.logits[:,-1:,:],
                    probs=probs.unsqueeze(1)
                )
                
                if config.store_mode == "memory":
                    generation_internals.append(step_internals)
                elif config.store_mode == "disk":
                    if step % config.save_every_n_steps == 0:
                        results.save_step_internals(
                            run_idx=run_idx,
                            step_idx=step,
                            internals=step_internals.to_numpy()
                        )
                
                # Cleanup for disk/off modes
                if config.store_mode in ["disk", "off"]:
                    del step_internals
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error during generation step {step}: {str(e)}", exc_info=True)
                raise
    
    logger.info(f"Generation completed. Reason: {stopping_reason or 'Maximum tokens reached'}")
    
    # Ensure output consistency
    while len(generated_texts) < len(generation_acts):
        generated_texts.append(tokenizer.decode(current_ids[0], skip_special_tokens=True))
    
    # Clean up KV cache
    token_gen.reset_cache()
    
    return generation_internals, generated_texts, current_ids, stopping_reason or "Maximum tokens reached"
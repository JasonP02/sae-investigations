import logging
from typing import List, Tuple, Set
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from config import GenerationConfig
from models import ModelInternals, ExperimentResults
from einops import rearrange
import numpy as np

# Configure logging once at module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logging

FILLER_PATTERNS: List[str] = [
    "is a", "is an", "is the", "and the", "then the",
    "to the", "of the", "in the", "on the", "at the",
    "and then", "so the", "which is", "that is"
]

PHRASE_END_TOKENS: Set[str] = {'.', '!', '?', ',', ';', ':'}

def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: Sae,
    input_text: str,
    results: ExperimentResults,
    run_idx: int,
    config: GenerationConfig = None,
) -> Tuple[List[ModelInternals], List[str], torch.Tensor, str]:

    

    inputs = tokenizer(input_text, return_tensors="pt")
    logger.debug(f"Tokenization successful. Shape: {inputs['input_ids'].shape}")
    

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    logger.debug("Successfully moved tokens to device")
    
    current_ids = inputs["input_ids"]
    logger.debug(f"Current IDs shape: {current_ids.shape}")

    
    generation_acts = []
    generated_texts = []
    recent_phrases = []
    recent_tokens = torch.zeros(config.repetition_window, dtype=torch.long, device=current_ids.device)
    consecutive_fillers = 0
    current_phrase = ""
    stopping_reason = None
    generation_internals = []

    logger.info("Beginning generation loop")
    with torch.inference_mode():
        for step in range(config.max_new_tokens):
            logger.debug(f"\nStep {step} ----")
            logger.debug(f"Current input shape: {current_ids.shape}")
            
            try:
                outputs = model(
                    current_ids, 
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=False  # We don't need caching since we're not using KQV
                )

                logger.debug(f"Model output logits shape: {outputs.logits.shape}")
                logger.debug(f"Number of hidden states: {len(outputs.hidden_states)}")
                logger.debug(f"Hidden state 10 shape: {outputs.hidden_states[10].shape}")
                
                # Check for NaN in logits
                if torch.isnan(outputs.logits).any():
                    logger.error("NaN detected in logits")
                    raise RuntimeError("Model produced NaN logits")
                
                next_token_logits = outputs.logits[:, -1:, :].squeeze(1)
                
                # Apply temperature scaling
                if config.do_sample:
                    next_token_logits = next_token_logits / config.temperature
                    
                # Apply softmax with numerical stability
                next_token_logits = next_token_logits - next_token_logits.max(dim=-1, keepdim=True)[0]
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Verify probabilities
                if torch.isnan(probs).any() or (probs < 0).any() or (probs > 1).any():
                    logger.error("Invalid probabilities detected")
                    raise RuntimeError("Invalid probability distribution")
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                logger.debug(f"Selected token: {next_token.item()}")
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
                token_text = tokenizer.decode([next_token[0].item()])
                logger.debug(f"Generated token text: '{token_text}'")
                
                current_phrase += token_text
                

                # Log SAE encoding
                sae_input = outputs.hidden_states[10]
                encoded = sae.encode(sae_input)  # Returns (top_acts, top_indices)
                logger.debug(f"SAE encoded top acts shape: {encoded[0].shape}")
                logger.debug(f"SAE encoded top indices shape: {encoded[1].shape}")
                
                step_acts = [{
                    'original': outputs.hidden_states[10].detach(),
                    'encoded_acts': encoded[0],
                    'encoded_indices': encoded[1]
                }]
                generation_acts.append(step_acts)

                if step % 5 == 0 or step == config.max_new_tokens - 1:
                    current_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
                    generated_texts.append(current_text)
                    logger.debug(f"Current full text: {current_text}")

                # Update recent tokens tracking
                recent_tokens[step % config.repetition_window] = next_token[0]
                
                # Check for stopping conditions
                if next_token[0].item() == (config.eos_token_id or tokenizer.eos_token_id):
                    logger.info("Stopping: EOS token generated")
                    stopping_reason = "EOS token generated"
                    break

                # Phrase similarity checking
                if any(token_text.endswith(end_token) for end_token in config.phrase_end_tokens):
                    logger.debug(f"End of phrase detected: '{current_phrase.strip()}'")
                    if current_phrase.strip():
                        recent_phrases.append(current_phrase.strip())
                        if len(recent_phrases) > config.max_recent_phrases:
                            recent_phrases.pop(0)
                    current_phrase = ""

                    if len(recent_phrases) >= 2:
                        words1 = set(recent_phrases[-2].lower().split())
                        words2 = set(recent_phrases[-1].lower().split())
                        if words1 and words2:
                            similarity = len(words1.intersection(words2)) / min(len(words1), len(words2))
                            logger.debug(f"Phrase similarity: {similarity:.2f}")
                            if similarity > config.semantic_similarity_threshold:
                                logger.info(f"Stopping: High semantic similarity ({similarity:.2f})")
                                stopping_reason = f"High semantic similarity ({similarity:.2f})"
                                break

                # Check for filler patterns
                current_text = tokenizer.decode(current_ids[0])
                if any(pattern in current_text[-config.phrase_context_window:].lower() for pattern in config.filler_patterns):
                    consecutive_fillers += 1
                    logger.debug(f"Filler pattern detected. Count: {consecutive_fillers}")
                    if consecutive_fillers > config.max_consecutive_fillers:
                        logger.info("Stopping: Too many consecutive filler phrases")
                        stopping_reason = "Too many consecutive filler phrases"
                        break
                else:
                    consecutive_fillers = 0

                # Check token diversity
                if step >= config.repetition_window:
                    unique_tokens = torch.unique(recent_tokens)
                    unique_ratio = len(unique_tokens) / config.repetition_window
                    logger.debug(f"Unique token ratio: {unique_ratio:.2f}")
                    if unique_ratio < config.min_unique_ratio:
                        logger.info(f"Stopping: Low unique token ratio ({unique_ratio:.2f})")
                        stopping_reason = f"Low unique token ratio ({unique_ratio:.2f})"
                        break

                # Store the step internals
                step_internals = ModelInternals(
                    hidden_states=outputs.hidden_states,
                    attention={
                        i: outputs.attentions[i]
                        for i in range(len(outputs.attentions))
                    },
                    mlp={
                        i: {
                            'pre_activation': outputs.hidden_states[i],
                            'post_activation': outputs.hidden_states[i+1]
                        }
                        for i in range(len(outputs.hidden_states)-1)
                    },
                    residual=outputs.hidden_states,
                    token=next_token[0].item(),
                    token_text=token_text,
                    logits=outputs.logits[:,-1:,:],
                    probs=torch.nn.functional.softmax(outputs.logits[:,-1:,:], dim=-1)
                )

                # Convert to numpy and save immediately
                numpy_internals = step_internals.to_numpy()
                results.save_step_internals(run_idx=run_idx, step_idx=step, internals=numpy_internals)

                # Clear CUDA memory
                del step_internals
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error during generation step {step}: {str(e)}", exc_info=True)
                raise

    logger.info(f"Generation completed. Reason: {stopping_reason or 'Maximum tokens reached'}")
    
    # Ensure generated_texts matches generation_acts length
    while len(generated_texts) < len(generation_acts):
        generated_texts.append(tokenizer.decode(current_ids[0], skip_special_tokens=True))
    
    logger.debug(f"Final output lengths - internals: {len(generation_internals)}, texts: {len(generated_texts)}")
    
    return generation_internals, generated_texts, current_ids, stopping_reason or "Maximum tokens reached"
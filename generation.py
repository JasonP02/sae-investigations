from typing import List, Tuple, Set
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from config import GenerationConfig

import logging
logging.basicConfig(
    filename="generation.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s"
)

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
    config: GenerationConfig = None,
) -> Tuple[List, List, torch.Tensor, str]:
    if config is None:
        config = GenerationConfig.default()

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    current_ids = inputs["input_ids"]
    generation_acts = []
    generated_texts = []
    recent_phrases = []
    recent_tokens = torch.zeros(config.repetition_window, dtype=torch.long, device=model.device)
    consecutive_fillers = 0
    current_phrase = ""
    stopping_reason = None

    with torch.inference_mode():
        for step in range(config.max_new_tokens):
            logging.debug(f"\nStarting step {step}")
            logging.debug(f"Current_ids device: {current_ids.device}")
            logging.debug(f"Current_ids dtype: {current_ids.dtype}")
            
            outputs = model(current_ids, output_hidden_states=True)
            logging.debug("Model output complete")
            logging.debug(f"Logits shape: {outputs.logits.shape}")
            
            next_token_logits = outputs.logits[:, -1:, :]
            logging.debug(f"Next token logits shape: {next_token_logits.shape}")
            logging.debug(f"Next token logits device: {next_token_logits.device}")

            if config.do_sample:
                logging.debug("Entered do_sample")
                next_token_logits = next_token_logits / config.temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                logging.debug(f"Probs shape: {probs.shape}")
                logging.debug(f"Probs max: {probs.max().item()}")
                
                max_prob = probs.max(dim=-1).values.max().item()
                if max_prob < config.min_confidence:
                    stopping_reason = f"Low confidence ({max_prob:.3f})"
                    break

                cumsum_probs = torch.cumsum(probs, dim=-1)
                mask = cumsum_probs > config.top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                probs[mask] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs[0, 0], 1).view(1, 1)
            else:
                logging.debug("Entered else")
                next_token = next_token_logits[0, 0].argmax().view(1, 1)

            logging.debug("Pre-cat")

            logging.debug("Current ids: %s", current_ids)
            logging.debug("Next token: %s", next_token)
            logging.debug("Current_ids shape: %s", current_ids.shape)
            logging.debug("Next token shape: %s", next_token.shape)

            # After line 93 (after next_token = torch.multinomial...)
            logging.debug(f"Step {step}: Generated next token")

            current_ids = torch.cat([current_ids, next_token], dim=1)
            logging.debug(f"Step {step}: Updated current_ids")
            
            token_text = tokenizer.decode([next_token[0, 0].item()])
            logging.debug(f"Step {step}: Decoded token text: {token_text}")
            
            current_phrase += token_text
            logging.debug(f"Step {step}: Updated current_phrase")
            
            step_acts = [{
                'original': outputs.hidden_states[10].detach(),
                'encoded': sae.encode(outputs.hidden_states[10])
            }]
            logging.debug(f"Step {step}: Created step_acts")
            
            generation_acts.append(step_acts)
            logging.debug(f"Step {step}: Added to generation_acts")

            if step % 5 == 0 or step == config.max_new_tokens - 1:
                generated_texts.append(tokenizer.decode(current_ids[0], skip_special_tokens=True))
                logging.debug(f"Step {step}: Updated generated_texts")

            recent_tokens[step % config.repetition_window] = next_token[0, 0]
            logging.debug(f"Step {step}: Updated recent_tokens")

            if next_token[0, 0].item() == (config.eos_token_id or tokenizer.eos_token_id):
                logging.debug(f"Step {step}: Found EOS token")
                stopping_reason = "EOS token generated"
                break

            if any(token_text.endswith(end_token) for end_token in config.phrase_end_tokens):
                logging.debug(f"Step {step}: Found phrase end token")
                if current_phrase.strip():
                    recent_phrases.append(current_phrase.strip())
                    if len(recent_phrases) > config.max_recent_phrases:
                        recent_phrases.pop(0)
                current_phrase = ""
                logging.debug(f"Step {step}: Updated recent_phrases")

                if len(recent_phrases) >= 2:
                    words1 = set(recent_phrases[-2].lower().split())
                    words2 = set(recent_phrases[-1].lower().split())
                    if words1 and words2:
                        similarity = len(words1.intersection(words2)) / min(len(words1), len(words2))
                        logging.debug(f"Step {step}: Calculated similarity: {similarity}")
                        if similarity > config.semantic_similarity_threshold:
                            stopping_reason = f"High semantic similarity ({similarity:.2f})"
                            break

            current_text = tokenizer.decode(current_ids[0])
            logging.debug(f"Step {step}: Decoded current text")
            
            if any(pattern in current_text[-config.phrase_context_window:].lower() for pattern in config.filler_patterns):
                consecutive_fillers += 1
                logging.debug(f"Step {step}: Incremented consecutive_fillers to {consecutive_fillers}")
                if consecutive_fillers > config.max_consecutive_fillers:
                    stopping_reason = "Too many consecutive filler phrases"
                    break
            else:
                consecutive_fillers = 0
                logging.debug(f"Step {step}: Reset consecutive_fillers")

            if step >= config.repetition_window:
                unique_tokens = torch.unique(recent_tokens)
                unique_ratio = len(unique_tokens) / config.repetition_window
                logging.debug(f"Step {step}: Calculated unique_ratio: {unique_ratio}")
                if unique_ratio < config.min_unique_ratio:
                    stopping_reason = f"Low unique token ratio ({unique_ratio:.2f})"
                    break

    while len(generated_texts) < len(generation_acts):
        generated_texts.append(tokenizer.decode(current_ids[0], skip_special_tokens=True))

    return generation_acts, generated_texts, current_ids[0], stopping_reason or "Maximum tokens reached" 
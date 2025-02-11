"""
Core experiment functionality for running SAE analysis.
"""

import torch
from collections import Counter
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
import gc
import traceback

from config import GenerationConfig
from models import ModelState, ExperimentResults
from generation import generate_text
from visualization import visualize_generation_activations
from setup import setup_model_and_sae

def run_multiple_experiments(
    prompt: str,
    num_runs: int,
    model_state: Optional[ModelState] = None,
    config: Optional[GenerationConfig] = None,
    device: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> ExperimentResults:
    """
    Run multiple generation experiments and collect statistics.
    
    Args:
        prompt: The input text to generate from
        num_runs: Number of times to run the experiment
        model_state: Pre-initialized ModelState (if None, will create new one)
        config: Generation configuration (uses default if None)
        device: Device to use (uses CUDA if available when None)
        progress_callback: Optional callback function to update progress
    
    Returns:
        ExperimentResults containing all experiment data and statistics
    """
    
    # Store results
    all_texts = []
    all_tokens = []
    stopping_reasons = Counter()
    generation_lengths = []
    all_generation_acts = []
        
    # Run experiments
    for i in range(num_runs):
        try:
            print(f"\nStarting run {i}...", flush=True)
            print("Config:", config, flush=True)
            print("Model device:", model_state.model.device, flush=True)
            print("SAE device:", model_state.sae.device, flush=True)
            print("Input prompt:", prompt, flush=True)
            
            with torch.no_grad():  # Add no_grad context
                # Run generation
                print("About to call generate_text...", flush=True)
                gen_acts, gen_texts, tokens, stopping_reason = generate_text(
                    model=model_state.model,
                    tokenizer=model_state.tokenizer,
                    sae=model_state.sae,
                    input_text=prompt,
                    config=config
                )
                print("Finished generate_text call", flush=True)
                
                # Add the stopping reason to the Counter
                if stopping_reason:
                    stopping_reasons[stopping_reason] += 1
                
                # Process results immediately and clear original data
                final_text = gen_texts[-1][len(prompt):]
                all_texts.append(final_text)
                all_tokens.extend(model_state.tokenizer.encode(final_text))
                generation_lengths.append(len(final_text.split()))
                
                # Store generation acts
                all_generation_acts.append(gen_acts)
                
                # Cleanup - remove reference to non-existent processed_acts
                del gen_acts, gen_texts, tokens
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"Error in run {i}: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        
        if progress_callback:
            progress_callback()
    
    # Calculate statistics
    token_frequencies = Counter(all_tokens)
    avg_length = sum(generation_lengths) / len(generation_lengths)
    
    # Calculate unique token ratio
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    unique_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    # Map token IDs to text for readability
    token_frequencies_text = Counter({
        model_state.tokenizer.decode([token_id]): count 
        for token_id, count in token_frequencies.most_common(20)
    })
    
    return ExperimentResults(
        model_state=model_state,
        config=config,
        prompt=prompt,
        all_texts=all_texts,
        stopping_reasons=stopping_reasons,
        token_frequencies=token_frequencies_text,
        avg_length=avg_length,
        unique_ratio=unique_ratio,
        generation_acts=all_generation_acts,
        metadata={'num_runs': num_runs}
    ) 

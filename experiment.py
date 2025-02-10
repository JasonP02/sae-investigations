"""
Core experiment functionality for running SAE analysis.
"""

import torch
from collections import Counter
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae # type: ignore
import gc
import traceback
from config import GenerationConfig
from models import ModelState, ExperimentResults
from generation import analyze_generation
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
    print("Starting run_multiple_experiments")  # Debug print 1
    
    if model_state is None:
        print("Setting up model and SAE...")  # Debug print 2
        model_state = setup_model_and_sae(device)
    
    if config is None:
        config = GenerationConfig.default()
    
    # Store results
    all_texts = []
    all_tokens = []
    stopping_reasons = Counter()
    generation_lengths = []
    all_generation_acts = []
    
    print(f"Beginning {num_runs} runs...")  # Debug print 3
    
    for i in range(num_runs):
        print(f"\nStarting run {i+1}/{num_runs}")  # Debug print 4
        with torch.no_grad():  # Add no_grad context
            print("Before analyze_generation call")  # Debug print 5
            # Run generation
            gen_acts, gen_texts, tokens, stopping_reason = analyze_generation(
                model=model_state.model,
                tokenizer=model_state.tokenizer,
                sae=model_state.sae,
                input_text=prompt,
                config=config
            )
            print("After analyze_generation call")  # Debug print 6
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

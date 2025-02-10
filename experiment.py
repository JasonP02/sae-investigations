"""
Core experiment functionality for running SAE analysis.
"""

import torch
from collections import Counter
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae # type: ignore

from config import GenerationConfig
from generation import analyze_generation
from visualization import visualize_generation_activations
from setup import setup_model_and_sae

def run_generation_experiment(
    prompt: str,
    config: GenerationConfig = None,
    device: str = None,
    model: AutoModelForCausalLM = None,
    tokenizer: AutoTokenizer = None,
    sae: Sae = None,
    visualize: bool = True
) -> None:
    """
    Run a complete generation experiment with visualization.
    
    Args:
        prompt: The input text to generate from
        config: Generation configuration (uses default if None)
        device: Device to use (uses CUDA if available when None)
        model: Pre-initialized model (if None, will create new one)
        tokenizer: Pre-initialized tokenizer (if None, will create new one)
        sae: Pre-initialized SAE (if None, will create new one)
        visualize: Whether to display visualization plots (default: True)
    """
    # Setup if components not provided
    if model is None or tokenizer is None or sae is None:
        model, tokenizer, sae = setup_model_and_sae(device)
    
    # Run generation
    gen_acts, gen_texts, tokens = analyze_generation(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        input_text=prompt,
        config=config
    )
    
    # Print results
    print("\nGeneration steps:", len(gen_texts))
    print(gen_texts[-1])
    
    # Visualize
    if visualize:
        figures = visualize_generation_activations(gen_acts, gen_texts)
        for fig in figures:
            fig.show() 

def run_multiple_experiments(
    prompt: str,
    num_runs: int,
    config: GenerationConfig = None,
    device: str = None,
    model: AutoModelForCausalLM = None,
    tokenizer: AutoTokenizer = None,
    sae: Sae = None,
) -> Dict:
    """
    Run multiple generation experiments and collect statistics.
    
    Args:
        prompt: The input text to generate from
        num_runs: Number of times to run the experiment
        config: Generation configuration (uses default if None)
        device: Device to use (uses CUDA if available when None)
        model: Pre-initialized model (if None, will create new one)
        tokenizer: Pre-initialized tokenizer (if None, will create new one)
        sae: Pre-initialized SAE (if None, will create new one)
    
    Returns:
        Dictionary containing:
        - all_texts: List of all generated texts
        - stopping_reasons: Counter of early stopping reasons
        - token_frequencies: Counter of most common tokens
        - avg_length: Average length of generations
        - unique_ratio: Average ratio of unique tokens
    """
    if model is None or tokenizer is None or sae is None:
        model, tokenizer, sae = setup_model_and_sae(device)
    
    if config is None:
        config = GenerationConfig.default()
    
    # Store results
    all_texts = []
    all_tokens = []
    stopping_reasons = Counter()
    generation_lengths = []
    
    # Run experiments
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}")
        
        # Capture output to parse stopping reason
        import io
        import sys
        output_capture = io.StringIO()
        sys.stdout = output_capture
        
        # Run generation
        gen_acts, gen_texts, tokens = analyze_generation(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            input_text=prompt,
            config=config
        )
        
        # Restore stdout and get captured output
        sys.stdout = sys.__stdout__
        output = output_capture.getvalue()
        
        # Parse stopping reason
        for line in output.split('\n'):
            if line.startswith('Stopping:'):
                reason = line.split(':', 1)[1].strip()
                stopping_reasons[reason] += 1
                break
        
        # Collect results
        final_text = gen_texts[-1][len(prompt):]  # Remove prompt
        all_texts.append(final_text)
        all_tokens.extend(tokenizer.encode(final_text))
        generation_lengths.append(len(final_text.split()))
    
    # Calculate statistics
    token_frequencies = Counter(all_tokens)
    avg_length = sum(generation_lengths) / len(generation_lengths)
    
    # Calculate unique token ratio
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    unique_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    # Map token IDs to text for readability
    token_frequencies_text = Counter({
        tokenizer.decode([token_id]): count 
        for token_id, count in token_frequencies.most_common(20)
    })
    
    return {
        'all_texts': all_texts,
        'stopping_reasons': stopping_reasons,
        'token_frequencies': token_frequencies_text,
        'avg_length': avg_length,
        'unique_ratio': unique_ratio
    } 

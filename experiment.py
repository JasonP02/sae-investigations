"""
Core experiment functionality for running SAE analysis.
"""

import torch
from collections import Counter
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae # type: ignore

from config import GenerationConfig, ModelState, ExperimentResults
from generation import analyze_generation
from visualization import visualize_generation_activations
from setup import setup_model_and_sae

def run_generation_experiment(
    prompt: str,
    model_state: Optional[ModelState] = None,
    config: Optional[GenerationConfig] = None,
    device: Optional[str] = None,
    visualize: bool = True
) -> Tuple[List, List[str], torch.Tensor]:
    """
    Run a complete generation experiment with visualization.
    
    Args:
        prompt: The input text to generate from
        model_state: Pre-initialized ModelState (if None, will create new one)
        config: Generation configuration (uses default if None)
        device: Device to use (uses CUDA if available when None)
        visualize: Whether to display visualization plots (default: True)
    
    Returns:
        Tuple of (generation_acts, generated_texts, tokens)
    """
    # Setup if components not provided
    if model_state is None:
        model_state = setup_model_and_sae(device)
    
    # Run generation
    gen_acts, gen_texts, tokens = analyze_generation(
        model=model_state.model,
        tokenizer=model_state.tokenizer,
        sae=model_state.sae,
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
            
    return gen_acts, gen_texts, tokens

def run_multiple_experiments(
    prompt: str,
    num_runs: int,
    model_state: Optional[ModelState] = None,
    config: Optional[GenerationConfig] = None,
    device: Optional[str] = None,
) -> ExperimentResults:
    """
    Run multiple generation experiments and collect statistics.
    
    Args:
        prompt: The input text to generate from
        num_runs: Number of times to run the experiment
        model_state: Pre-initialized ModelState (if None, will create new one)
        config: Generation configuration (uses default if None)
        device: Device to use (uses CUDA if available when None)
    
    Returns:
        ExperimentResults containing all experiment data and statistics
    """
    if model_state is None:
        model_state = setup_model_and_sae(device)
    
    if config is None:
        config = GenerationConfig.default()
    
    # Store results
    all_texts = []
    all_tokens = []
    stopping_reasons = Counter()
    generation_lengths = []
    all_generation_acts = []
    
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
            model=model_state.model,
            tokenizer=model_state.tokenizer,
            sae=model_state.sae,
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
        all_tokens.extend(model_state.tokenizer.encode(final_text))
        generation_lengths.append(len(final_text.split()))
        all_generation_acts.append(gen_acts)
    
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

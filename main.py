"""
Main script for running SAE analysis experiments.
This script provides a clean interface for running generation experiments
with different configurations and visualizing the results.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae

from .config import GenerationConfig
from .generation import analyze_generation
from .visualization import visualize_generation_activations
from .setup import setup_model_and_sae

def run_generation_experiment(
    prompt: str,
    config: GenerationConfig = None,
    device: str = None,
    model: AutoModelForCausalLM = None,
    tokenizer: AutoTokenizer = None,
    sae: Sae = None
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
    print("Final text:", gen_texts[-1])
    
    # Visualize
    figures = visualize_generation_activations(gen_acts, gen_texts)
    for fig in figures:
        fig.show()

if __name__ == "__main__":
    # Example usage
    prompt = "Answer the following question: Q: What will happen if a ball is thrown at a wall? A:"
    
    # Use creative configuration
    config = GenerationConfig.creative()
    
    # Run experiment
    run_generation_experiment(prompt, config) 
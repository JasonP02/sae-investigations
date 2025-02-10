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

def setup_model_and_sae(device: str = None) -> tuple:
    """
    Initialize the model, tokenizer, and SAE with appropriate settings.
    
    Args:
        device: Device to use ('cuda' or 'cpu'). If None, will use CUDA if available.
    
    Returns:
        tuple of (model, tokenizer, sae)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize SAE and tokenizer
    sae = Sae.load_from_hub(
        "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k", 
        hookpoint="layers.10.mlp",
        device=device
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device_map={"": device},
        torch_dtype=torch.float16,  # Use half precision for 6GB VRAM
        low_cpu_mem_usage=True,
        max_memory={0: "5GB"},  # Reserve some VRAM for other operations
        use_flash_attention_2=False,  # RTX 2060 doesn't support it
    ).to(device)
    
    # Try to compile model for speed
    try:
        print("Attempting to compile model for speed...")
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=False,
            options={
                "max_autotune": False,
                "max_parallel_gemm": 1,  # More conservative for 6GB VRAM
                "triton.cudagraphs": False,  # Disable cudagraphs for older GPUs
            }
        )
        print("Model compiled successfully!")
    except Exception as e:
        print(f"Could not compile model (requires PyTorch 2.0+): {e}")
        print("Continuing without compilation...")
    
    # Move SAE to half precision for consistency
    sae = sae.to(torch.float16)
    
    return model, tokenizer, sae

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
"""
Setup utilities for SAE analysis experiments.
This module provides functions for initializing models and components.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from models import ModelState

def setup_model_and_sae(device: str = None) -> ModelState:
    """
    Initialize the model, tokenizer, and SAE with appropriate settings.
    
    Args:
        device: Device to use ('cuda' or 'cpu'). If None, will use CUDA if available.
    
    Returns:
        ModelState containing initialized components
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    sae_name = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
    
    # Initialize SAE and tokenizer
    sae = Sae.load_from_hub(
        sae_name,
        hookpoint="layers.10.mlp",
        device=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with optimizations but without compilation
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        torch_dtype=torch.float16,  # Use half precision for 6GB VRAM
        low_cpu_mem_usage=True,
        max_memory={0: "5GB"},  # Reserve some VRAM for other operations
        use_flash_attention_2=False,  # RTX 2060 doesn't support it
    ).to(device)
    
    # Move SAE to half precision for consistency
    sae = sae.to(torch.float16)
    
    return ModelState(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        device=device,
        model_name=model_name,
        sae_name=sae_name
    ) 
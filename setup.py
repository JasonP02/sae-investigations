import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from models import ModelState
import traceback

def setup_model_and_sae(device: str = None, enable_compile: bool = True) -> ModelState:
    """
    Initialize the model, tokenizer, and SAE with settings tuned for RTX 2060.
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

    # Load model with optimizations for limited VRAM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        torch_dtype=torch.float16,      # Use half precision to reduce VRAM usage
        low_cpu_mem_usage=True,
        max_memory={0: "5GB"},            # Reserve VRAM for other operations
        use_flash_attention_2=False,      # RTX 2060 does not support Flash Attention 2
    ).to(device)

    try:
        print("Optimizing model for RTX 2060 inference...")
        
        # Core PyTorch optimizations
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
        torch.backends.cudnn.benchmark = True         # Enable cudnn autotuner
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # Enable memory-efficient attention
        
        # If compilation is enabled (though often faster without for RTX 2060)
        if enable_compile:
            # Conservative compilation settings that work well for RTX 2060
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = False
            torch._inductor.config.max_autotune = True
            torch._inductor.config.coordinate_descent_tuning = False
            torch._inductor.config.triton.unique_kernel_names = True
            torch._inductor.config.optimize_execution_layout = True
            torch._inductor.config.force_fuse_diagonal_fusion = True
            
            model = torch.compile(
                model,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=True,
                options={
                    "max_autotune": True,
                    "trace.enabled": True,
                    "trace.graph_diagram": False,
                    "max_parallel_blocks": 2,  # Limit parallel blocks for 6GB VRAM
                }
            )
            
            print("Testing compiled model...")
            test_input = tokenizer("Test input", return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                _ = model(test_input)
            print("Model compiled successfully!")
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        print("Continuing with base model...")

    # Memory optimizations
    if device == "cuda":
        torch.cuda.empty_cache()
        # Set optimal CUDA memory allocation settings
        torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve small buffer for CUDA
        torch.cuda.memory.set_per_process_memory_fraction(0.95)
    
    # Convert model and SAE to half precision
    model = model.half()  # Use FP16 for faster inference
    sae = sae.half()

    return ModelState(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        device=device,
        model_name=model_name,
        sae_name=sae_name
    )

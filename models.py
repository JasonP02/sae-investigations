"""
Model state and experiment result containers.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae # type: ignore
from config import GenerationConfig
import torch
import numpy as np
import time
from datetime import datetime
import os
import json

@dataclass
class ModelState:
    """
    Container for model, tokenizer, and SAE state.
    
    Attributes:
        model: The language model
        tokenizer: The tokenizer
        sae: The sparse autoencoder
        device: The device the model is on
        model_name: Name/identifier of the model
        sae_name: Name/identifier of the SAE
    """
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    sae: Sae
    device: str
    model_name: str
    sae_name: str
    
    @property
    def components(self) -> tuple:
        """Returns model components as a tuple for easy unpacking."""
        return self.model, self.tokenizer, self.sae


@dataclass
class ModelInternals:
    """Container for model's internal states during generation."""
    hidden_states: List[torch.Tensor]
    attention: Dict[int, torch.Tensor]
    mlp: Dict[int, Dict[str, torch.Tensor]]
    residual: List[torch.Tensor]
    token: int
    token_text: str
    logits: torch.Tensor
    probs: torch.Tensor
    
    def to_numpy(self, save_layers: Set[int] = None) -> Dict[str, Any]:
        """Convert tensors to numpy arrays efficiently."""
        if save_layers is None:
            save_layers = {10}
            
        def safe_convert(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().to(torch.float32).numpy()
        
        # Save layer 10 data including attention patterns and residual stream
        return {
            'layer_10': {  # Wrap in layer_10 key
                'mlp_pre': safe_convert(self.mlp[10]['pre_activation']),
                'mlp_post': safe_convert(self.mlp[10]['post_activation']),
                'hidden': safe_convert(self.hidden_states[10]),
                'attention': safe_convert(self.attention[10]),
                'residual': safe_convert(self.residual[10]),
                'logits': safe_convert(self.logits),
                'probs': safe_convert(self.probs),
                'token': self.token,
                'token_text': self.token_text
            }
        }

@dataclass
class ExperimentResults:
    """Container for experiment results and analysis."""
    model_state: ModelState
    config: GenerationConfig
    prompt: str
    all_texts: List[str] = field(default_factory=list)
    stopping_reasons: Counter = field(default_factory=Counter)
    token_frequencies: Counter = field(default_factory=Counter)
    avg_length: float = 0.0
    unique_ratio: float = 0.0
    
    def __post_init__(self):
        """Setup experiment directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = timestamp
        self.experiment_path = f"experiments/{timestamp}"
        
        # Create directory structure
        for subdir in ['metadata', 'runs', 'stats']:
            os.makedirs(f"{self.experiment_path}/{subdir}", exist_ok=True)
            
        # Save experiment metadata
        self._save_metadata()
    
    def _save_metadata(self):
        """Save experiment configuration and metadata."""
        metadata = {
            'timestamp': self.experiment_id,
            'model': self.model_state.model_name,
            'sae': self.model_state.sae_name,
            'prompt': self.prompt,
            'config': self.config.to_dict()
        }
        
        with open(f"{self.experiment_path}/metadata/config.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_step_internals(self, run_idx: int, step_idx: int, internals: Dict):
        """Save step data efficiently."""
        run_dir = f"{self.experiment_path}/runs/run_{run_idx:03d}"
        os.makedirs(run_dir, exist_ok=True)
        
        # Save as compressed numpy
        np.savez_compressed(
            f"{run_dir}/step_{step_idx:04d}.npz",
            **internals['layer_10']  # Only save layer 10 data
        )

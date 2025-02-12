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
import shutil
import logging

logger = logging.getLogger(__name__)

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
    """Container for experiment results and metadata."""
    model_state: Optional[ModelState] = None
    config: Optional[GenerationConfig] = None
    prompt: str = ""
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    experiment_path: str = field(init=False)
    all_texts: List[str] = field(default_factory=list)
    all_tokens: List[int] = field(default_factory=list)
    generation_lengths: List[int] = field(default_factory=list)
    stopping_reasons: Counter = field(default_factory=Counter)
    generation_internals: Optional[List[List[ModelInternals]]] = None  # For memory mode
    
    @classmethod
    def clear_all_experiment_files(cls):
        """Clear all files in the experiments directory."""
        experiments_dir = "experiments"
        if os.path.exists(experiments_dir):
            logger.info(f"Clearing all experiment files in {experiments_dir}")
            try:
                shutil.rmtree(experiments_dir)
                os.makedirs(experiments_dir)  # Recreate empty directory
                logger.info("Successfully cleared all experiment files")
            except Exception as e:
                logger.error(f"Error clearing experiment files: {str(e)}")
                raise
        else:
            logger.warning(f"Experiments directory {experiments_dir} does not exist")
            os.makedirs(experiments_dir)  # Create it if it doesn't exist
    
    def __post_init__(self):
        """Setup experiment directory structure."""
        self.experiment_path = f"experiments/{self.experiment_id}"
        
        # Create directory structure
        for subdir in ['metadata', 'runs', 'stats']:
            os.makedirs(f"{self.experiment_path}/{subdir}", exist_ok=True)
            
        # Save experiment metadata
        self._save_metadata()
    
    def clear_experiment_files(self):
        """Clear all files in the experiment directory."""
        if os.path.exists(self.experiment_path):
            logger.info(f"Clearing experiment files in {self.experiment_path}")
            try:
                shutil.rmtree(self.experiment_path)
                os.makedirs(self.experiment_path)  # Recreate empty directory
                # Recreate subdirectories
                for subdir in ['metadata', 'runs', 'stats']:
                    os.makedirs(f"{self.experiment_path}/{subdir}", exist_ok=True)
                logger.info("Successfully cleared experiment files")
            except Exception as e:
                logger.error(f"Error clearing experiment files: {str(e)}")
                raise
        else:
            logger.warning(f"Experiment path {self.experiment_path} does not exist")
    
    def _save_metadata(self):
        """Save experiment configuration and metadata."""
        metadata = {
            'timestamp': self.experiment_id,
            'model': self.model_state.model_name if self.model_state else None,
            'sae': self.model_state.sae_name if self.model_state else None,
            'prompt': self.prompt,
            'config': self.config.to_dict() if self.config else None
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

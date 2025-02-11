"""
Model state and experiment result containers.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae # type: ignore
from config import GenerationConfig
import torch

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
class ExperimentResults:
    """
    Container for experiment results and analysis.
    
    Attributes:
        model_state: The model state used for the experiment
        config: The generation configuration used
        prompt: The input prompt used
        all_texts: List of all generated texts
        stopping_reasons: Counter of early stopping reasons
        token_frequencies: Counter of most common tokens
        avg_length: Average length of generations
        unique_ratio: Average ratio of unique tokens
        generation_internals: List of ModelInternals for each run
        metadata: Additional experiment metadata
    """
    model_state: ModelState
    config: GenerationConfig
    prompt: str
    all_texts: List[str]
    stopping_reasons: Counter
    token_frequencies: Counter
    avg_length: float
    unique_ratio: float
    generation_internals: Optional[List[List[ModelInternals]]] = None  # List[runs][steps]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary format for visualization."""
        return {
            'all_texts': self.all_texts,
            'stopping_reasons': self.stopping_reasons,
            'token_frequencies': self.token_frequencies,
            'avg_length': self.avg_length,
            'unique_ratio': self.unique_ratio,
            'generation_internals': [
                [step.to_numpy() for step in run] 
                for run in (self.generation_internals or [])
            ],
            'metadata': {
                'model_name': self.model_state.model_name,
                'sae_name': self.model_state.sae_name,
                'prompt': self.prompt,
                **self.metadata
            }
        }

@dataclass
class ModelInternals:
    """Container for model's internal states during generation.
    
    Attributes:
        hidden_states: List of tensors [batch, seq_len, hidden_dim] for each layer
        attention: Dict containing attention patterns for each layer
            - scores: Raw attention scores
            - patterns: Softmaxed attention weights
            - values: Value vectors
            - keys: Key vectors
            - queries: Query vectors
        mlp: Dict containing MLP internals for each layer
            - pre_activation: Input to MLP
            - post_activation: Output after activation
            - sae_encoded: SAE encoded version (only for layer 10)
        residual: Residual stream values at each layer
        token: The generated token at this step
        token_text: The text representation of the token
        logits: Raw logits for token prediction
        probs: Softmaxed probabilities
    """
    hidden_states: List[torch.Tensor]
    attention: Dict[int, Dict[str, torch.Tensor]]
    mlp: Dict[int, Dict[str, torch.Tensor]]
    residual: List[torch.Tensor]
    token: int
    token_text: str
    logits: torch.Tensor
    probs: torch.Tensor
    
    def to_numpy(self) -> Dict[str, Any]:
        """Convert tensors to numpy arrays for visualization."""
        return {
            'hidden_states': [h.detach().cpu().numpy() for h in self.hidden_states],
            'attention': {
                layer: {k: v.detach().cpu().numpy() 
                       for k, v in layer_data.items()}
                for layer, layer_data in self.attention.items()
            },
            'mlp': {
                layer: {k: v.detach().cpu().numpy() 
                       for k, v in layer_data.items()}
                for layer, layer_data in self.mlp.items()
            },
            'residual': [r.detach().cpu().numpy() for r in self.residual],
            'logits': self.logits.detach().cpu().numpy(),
            'probs': self.probs.detach().cpu().numpy()
        } 
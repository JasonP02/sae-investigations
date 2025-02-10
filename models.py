"""
Model state and experiment result containers.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae # type: ignore
from config import GenerationConfig

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
        generation_acts: Optional list of activation patterns
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
    generation_acts: Optional[List] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary format for visualization."""
        return {
            'all_texts': self.all_texts,
            'stopping_reasons': self.stopping_reasons,
            'token_frequencies': self.token_frequencies,
            'avg_length': self.avg_length,
            'unique_ratio': self.unique_ratio,
            'generation_acts': self.generation_acts,
            'metadata': {
                'model_name': self.model_state.model_name,
                'sae_name': self.model_state.sae_name,
                'prompt': self.prompt,
                **self.metadata
            }
        } 
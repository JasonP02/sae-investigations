"""
SAE Investigations Package

This package provides tools for analyzing and visualizing
the behavior of Sparse Autoencoders (SAE) during text generation.
"""

from .config import GenerationConfig
from .generation import analyze_generation
from .visualization import (
    visualize_generation_activations,
    create_activation_heatmap,
    create_feature_evolution_plot
)
from .main import setup_model_and_sae, run_generation_experiment

__version__ = "0.1.0"
__all__ = [
    'GenerationConfig',
    'analyze_generation',
    'visualize_generation_activations',
    'create_activation_heatmap',
    'create_feature_evolution_plot',
    'setup_model_and_sae',
    'run_generation_experiment'
] 
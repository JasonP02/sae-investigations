"""Visualization utilities for SAE analysis experiments.

This module provides visualization tools for analyzing Sparse Autoencoder (SAE)
experiments. It includes:
1. Generation statistics visualization (token frequencies, lengths, etc.)
2. Model internals visualization (attention, MLP activations, residual stream)
"""

from typing import List, Dict, Union, Optional
from collections import Counter
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import ExperimentResults, ModelInternals
from einops import rearrange

class StatisticsVisualizer:
    """Handles calculation and visualization of generation statistics."""
    
    def __init__(self, results: ExperimentResults):
        self.results = results
        self._calculate_statistics()
    
    def _calculate_statistics(self) -> None:
        """Calculate basic statistics from generation results."""
        # Token frequencies
        token_frequencies = Counter(self.results.all_tokens)
        
        # Average length
        self.results.avg_length = np.mean(self.results.generation_lengths)
        
        # Unique token ratio
        unique_tokens = len(set(self.results.all_tokens))
        total_tokens = len(self.results.all_tokens)
        self.results.unique_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # Map token IDs to text
        self.results.token_frequencies = Counter({
            self.results.model_state.tokenizer.decode([token_id]): count 
            for token_id, count in token_frequencies.most_common(20)
        })
    
    def plot_basic_stats(self) -> go.Figure:
        """Create basic statistics visualization."""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Average Length', 'Unique Token Ratio'],
            y=[self.results.avg_length, self.results.unique_ratio*100],
            text=[f"{self.results.avg_length:.1f}", f"{self.results.unique_ratio:.2%}"],
            textposition='auto',
        ))
        fig.update_layout(
            title="Generation Statistics",
            height=400, width=600,
            showlegend=False
        )
        return fig
    
    def plot_stopping_distribution(self) -> Optional[go.Figure]:
        """Create stopping reasons distribution visualization."""
        if not self.results.stopping_reasons:
            return None
            
        fig = go.Figure(data=[go.Pie(
            labels=list(self.results.stopping_reasons.keys()),
            values=list(self.results.stopping_reasons.values()),
            textinfo='label+percent'
        )])
        fig.update_layout(
            title="Early Stopping Distribution",
            height=400, width=600
        )
        return fig
    
    def create_samples_table(self) -> Optional[go.Figure]:
        """Create table of sample generations."""
        if not self.results.all_texts:
            return None
            
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Sample', 'Stopping Reason']),
            cells=dict(values=[
                self.results.all_texts,
                [self.results.stopping_reasons.most_common()[0][0]] * len(self.results.all_texts)
            ])
        )])
        fig.update_layout(
            title="Sample Generations",
            height=400, width=800
        )
        return fig

class ModelInternalsVisualizer:
    """Handles visualization of model's internal states."""
    
    def __init__(self, internals: List[ModelInternals]):
        self.internals = internals
    
    def plot_attention_patterns(self) -> List[go.Figure]:
        """Plot attention patterns across layers and heads."""
        figures = []
        for layer in range(len(self.internals[0].attention)):
            patterns = np.stack([
                step.attention[layer].cpu().numpy()
                for step in self.internals
            ])
            
            fig = make_subplots(
                rows=2, cols=4,  # 8 attention heads
                subplot_titles=[f"Head {i}" for i in range(8)],
                vertical_spacing=0.1
            )
            
            for head in range(8):
                fig.add_trace(
                    go.Heatmap(
                        z=patterns[:,:,head],
                        colorscale='RdBu',
                        showscale=head==0
                    ),
                    row=(head//4)+1, col=(head%4)+1
                )
            
            fig.update_layout(
                title=f"Layer {layer} Attention Patterns",
                height=800, width=1200
            )
            figures.append(fig)
        
        return figures
    
    def plot_mlp_activations(self) -> List[go.Figure]:
        """Plot MLP activation patterns for layer 10 (SAE layer)."""
        # Stack activations across steps
        original_acts = rearrange([
            internal.mlp[10]['pre_activation'][:, -1].cpu()
            for internal in self.internals
        ], 'step b d -> step d')
        
        encoded_acts = rearrange([
            internal.mlp[10]['sae_encoded'][:, -1].cpu()
            for internal in self.internals
        ], 'step b d -> step d')
        
        step_labels = [
            f"Step {step}: {internal.token_text}"
            for step, internal in enumerate(self.internals)
        ]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Original MLP', 'SAE-Encoded'),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Heatmap(
                z=original_acts.numpy(),
                y=step_labels,
                colorscale='RdBu',
                showscale=True,
                name='Original'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=encoded_acts.numpy(),
                y=step_labels,
                colorscale='RdBu',
                showscale=True,
                name='SAE-Encoded'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Layer 10 MLP Activation Patterns",
            height=800, width=1000
        )
        
        return [fig]
    
    def plot_residual_stream(self) -> List[go.Figure]:
        """Plot residual stream evolution."""
        # TODO: Implement residual stream visualization
        # This is a placeholder for future implementation
        return []

def visualize_experiment_results(results: ExperimentResults) -> None:
    """Create and display all visualizations for experiment results.
    
    Args:
        results: ExperimentResults containing experiment data
    """
    # Initialize visualizers
    stats_viz = StatisticsVisualizer(results)
    
    # Create and show statistics visualizations
    figures = []
    
    # Basic statistics
    basic_stats = stats_viz.plot_basic_stats()
    if basic_stats:
        figures.append(basic_stats)
    
    # Stopping distribution
    stopping_dist = stats_viz.plot_stopping_distribution()
    if stopping_dist:
        figures.append(stopping_dist)
    
    # Samples table
    samples_table = stats_viz.create_samples_table()
    if samples_table:
        figures.append(samples_table)
    
    # Show all figures
    for fig in figures:
        fig.show()
    
    # If we have internals data, show detailed visualizations
    if results.generation_internals:
        internals_viz = ModelInternalsVisualizer(results.generation_internals[0])
        
        # Attention patterns
        for fig in internals_viz.plot_attention_patterns():
            fig.show()
        
        # MLP activations
        for fig in internals_viz.plot_mlp_activations():
            fig.show()
        
        # Residual stream (when implemented)
        for fig in internals_viz.plot_residual_stream():
            fig.show()
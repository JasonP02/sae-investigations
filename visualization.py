"""Visualization utilities for SAE analysis experiments."""

from typing import List, Dict, Union
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import ExperimentResults, ModelInternals
from einops import rearrange

def visualize_experiment_results(results: ExperimentResults) -> List[go.Figure]:
    """Create visualizations for experiment-level results.
    
    Args:
        results: ExperimentResults containing experiment data
        
    Returns:
        List of plotly figures showing experiment statistics
    """
    figures = []
    
    # 1. Generation Statistics
    fig_stats = go.Figure()
    fig_stats.add_trace(go.Bar(
        x=['Average Length', 'Unique Token Ratio'],
        y=[results.avg_length, results.unique_ratio*100],
        text=[f"{results.avg_length:.1f}", f"{results.unique_ratio:.2%}"],
        textposition='auto',
    ))
    fig_stats.update_layout(
        title="Generation Statistics",
        height=400, width=600,
        showlegend=False
    )
    figures.append(fig_stats)
    
    # 2. Stopping Reasons
    if results.stopping_reasons:
        fig_stops = go.Figure(data=[go.Pie(
            labels=list(results.stopping_reasons.keys()),
            values=list(results.stopping_reasons.values()),
            textinfo='label+percent'
        )])
        fig_stops.update_layout(
            title="Early Stopping Distribution",
            height=400, width=600
        )
        figures.append(fig_stops)
    
    # 3. Sample Generations Table
    if results.all_texts:
        fig_samples = go.Figure(data=[go.Table(
            header=dict(values=['Sample', 'Stopping Reason']),
            cells=dict(values=[
                results.all_texts,
                [results.stopping_reasons.most_common()[0][0]] * len(results.all_texts)
            ])
        )])
        fig_samples.update_layout(
            title="Sample Generations",
            height=400, width=800
        )
        figures.append(fig_samples)
    
    for fig in figures:
        fig.show()

def plot_attention_patterns(internals: List[ModelInternals]) -> List[go.Figure]:
    """Plot attention patterns across layers and heads."""
    figures = []
    for layer in range(len(internals[0].attention)):
        patterns = np.stack([
            step.attention[layer]['patterns'][0].numpy()
            for step in internals
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

def plot_mlp_activations(internals: List[ModelInternals]) -> List[go.Figure]:
    """Plot MLP activation patterns for layer 10 (SAE layer)."""
    # Stack activations across steps
    original_acts = rearrange([
        internal.mlp[10]['pre_activation'][:, -1]
        for internal in internals
    ], 'step b d -> step d')
    
    encoded_acts = rearrange([
        internal.mlp[10]['sae_encoded'][:, -1]
        for internal in internals
    ], 'step b d -> step d')
    
    step_labels = [
        f"Step {step}: {internal.token_text}"
        for step, internal in enumerate(internals)
    ]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Original MLP', 'SAE-Encoded'),
        vertical_spacing=0.15
    )
    
    fig.add_trace(
        go.Heatmap(
            z=original_acts.cpu().numpy(),
            y=step_labels,
            colorscale='RdBu',
            showscale=True,
            name='Original'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=encoded_acts.cpu().numpy(),
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

def plot_residual_stream(internals: List[ModelInternals]) -> List[go.Figure]:
    """Plot residual stream evolution."""
    # TODO: Implement residual stream visualization
    return []
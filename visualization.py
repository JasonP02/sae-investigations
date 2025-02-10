from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go

def visualize_generation_activations(
    generation_acts: List,
    generated_texts: List[str],
    layer_idx: int = 0,
    step_interval: int = 1
) -> List[go.Figure]:
    """
    Visualize how activations change during the generation process.
    
    Creates two visualizations:
    1. A heatmap showing feature activations over generation steps
    2. A line plot showing the evolution of top feature activations
    
    Args:
        generation_acts: List of activation patterns for each generation step
        generated_texts: List of generated text at each step
        layer_idx: Index within each step's activations (should be 0 since we only collect one layer)
        step_interval: Show every nth step to reduce visual complexity
    
    Returns:
        List of plotly figures:
        - figures[0]: Feature activation heatmap
        - figures[1]: Top feature evolution plot
    """
    figures = []
    
    # 1. Feature activation heatmap
    fig_gen_heatmap = create_activation_heatmap(
        generation_acts,
        generated_texts,
        layer_idx,
        step_interval
    )
    figures.append(fig_gen_heatmap)
    
    # 2. Feature usage evolution
    fig_feature_evolution = create_feature_evolution_plot(
        generation_acts,
        generated_texts,
        layer_idx,
        step_interval
    )
    figures.append(fig_feature_evolution)
    
    return figures

def create_activation_heatmap(
    generation_acts: List,
    generated_texts: List[str],
    layer_idx: int,
    step_interval: int
) -> go.Figure:
    """Creates a heatmap of feature activations over generation steps."""
    fig = go.Figure()
    
    # Get activations for specified layer across all steps
    step_acts = []
    step_indices = []
    step_labels = []
    
    for step, (acts, text) in enumerate(zip(generation_acts, generated_texts)):
        if step % step_interval == 0:
            layer_acts = acts[0]
            step_acts.append(layer_acts.top_acts.detach().cpu().numpy()[0, -1])
            step_indices.append(layer_acts.top_indices.detach().cpu().numpy()[0, -1])
            step_labels.append(f"Step {step}: {text[-20:]}")
    
    # Convert to numpy arrays
    step_acts = np.array(step_acts)
    step_indices = np.array(step_indices)
    
    # Create heatmap
    fig.add_trace(
        go.Heatmap(
            z=step_acts,
            y=step_labels,
            colorscale='RdBu',
            name='Activation Values',
            showscale=True,
            colorbar=dict(title="Activation Value")
        )
    )
    
    fig.update_layout(
        title_text=f"Feature Activations During Generation (MLP Layer 10)",
        xaxis_title="K-index",
        yaxis_title="Generation Step",
        height=800,
        width=1200
    )
    
    return fig

def create_feature_evolution_plot(
    generation_acts: List,
    generated_texts: List[str],
    layer_idx: int,
    step_interval: int
) -> go.Figure:
    """Creates a line plot showing the evolution of top feature activations."""
    fig = go.Figure()
    
    # Get activations and indices
    step_acts = []
    step_indices = []
    
    for step, (acts, text) in enumerate(zip(generation_acts, generated_texts)):
        if step % step_interval == 0:
            layer_acts = acts[0]
            step_acts.append(layer_acts.top_acts.detach().cpu().numpy()[0, -1])
            step_indices.append(layer_acts.top_indices.detach().cpu().numpy()[0, -1])
    
    step_acts = np.array(step_acts)
    step_indices = np.array(step_indices)
    
    # Track top N most used features
    N = 10
    feature_counts = np.zeros(65536)
    for step_idx in step_indices:
        unique, counts = np.unique(step_idx, return_counts=True)
        feature_counts[unique] += counts
    
    top_features = np.argsort(feature_counts)[-N:]
    
    # Get the generated tokens for hover text
    all_tokens = []
    for text_idx in range(len(generated_texts)-1):
        if text_idx + 1 < len(generated_texts):
            new_token = generated_texts[text_idx + 1][len(generated_texts[text_idx]):]
            all_tokens.append(new_token if new_token else "")
    
    # Plot activation strength of top features
    for feature_idx in top_features:
        feature_activations = []
        hover_texts = []
        
        for step, (step_act, step_idx) in enumerate(zip(step_acts, step_indices)):
            feature_pos = np.where(step_idx == feature_idx)[0]
            activation = step_act[feature_pos[0]] if len(feature_pos) > 0 else 0
            feature_activations.append(activation)
            
            token_text = all_tokens[step-1] if step > 0 and step <= len(all_tokens) else ""
            hover_text = f"Step {step}<br>Feature: {feature_idx}<br>Activation: {activation:.3f}"
            if token_text:
                hover_text += f"<br>Token: '{token_text}'"
            hover_texts.append(hover_text)
        
        fig.add_trace(
            go.Scatter(
                y=feature_activations,
                name=f"Feature {feature_idx}",
                mode='lines+markers',
                hovertext=hover_texts,
                hoverinfo='text'
            )
        )
    
    fig.update_layout(
        title_text=f"Top {N} Feature Evolution During Generation (MLP Layer 10)",
        xaxis_title="Generation Step",
        yaxis_title="Activation Strength",
        height=600,
        width=1000,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    return fig 
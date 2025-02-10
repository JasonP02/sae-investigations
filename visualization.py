from typing import List, Tuple, Dict, Union
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import ExperimentResults

class BaseVisualizer:
    """Base class for all visualizers."""
    def __init__(self, results: Union[ExperimentResults, Dict]):
        self.results = results if isinstance(results, dict) else results.to_dict()
        self.base_title = self._create_base_title()
    
    def _create_base_title(self) -> str:
        """Create a base title using metadata if available."""
        title = "Generation Analysis Results"
        if 'metadata' in self.results:
            meta = self.results['metadata']
            if 'model_name' in meta:
                title += f" - {meta['model_name']}"
            if 'num_runs' in meta:
                title += f" ({meta['num_runs']} runs)"
        return title
    
    def create_figures(self) -> List[go.Figure]:
        """Create all visualization figures. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_figures()")
    
    def _validate_figure(self, fig: go.Figure) -> go.Figure:
        """Validate that a figure is a proper plotly Figure object."""
        if not isinstance(fig, go.Figure):
            raise ValueError(f"Expected plotly Figure object, got {type(fig)}")
        return fig

def create_activation_heatmaps(
    prepared_acts: List[Dict],
    generated_texts: List[str],
    title: str = "Feature Activations During Generation (MLP Layer 10)"
) -> go.Figure:
    """Creates heatmaps using pre-converted numpy arrays."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Original MLP Activations', 'SAE-Encoded Activations'),
        vertical_spacing=0.15
    )
    
    step_labels = [f"Step {i}: {generated_texts[i][-20:]}" for i in range(len(prepared_acts))]
    
    # Original activations heatmap
    original_acts = np.stack([acts['original'][0, -1] for acts in prepared_acts])
    fig.add_trace(
        go.Heatmap(z=original_acts, y=step_labels, colorscale='RdBu'),
        row=1, col=1
    )
    
    # SAE-encoded activations heatmap
    encoded_acts = np.stack([acts['encoded']['top_acts'][0, -1] for acts in prepared_acts])
    fig.add_trace(
        go.Heatmap(z=encoded_acts, y=step_labels, colorscale='RdBu'),
        row=2, col=1
    )
    
    return fig

def create_feature_evolution_plot(
    prepared_acts: List[Dict],
    generated_texts: List[str],
    n_features: int = 10,
    title: str = "Top Feature Evolution During Generation"
) -> go.Figure:
    """Creates evolution plot using pre-converted numpy arrays."""
    fig = go.Figure()
    n_features = int(n_features)
    
    # Get top N most active features
    encoded_indices = np.concatenate([acts['encoded']['top_indices'][0, -1] for acts in prepared_acts])
    feature_counts = np.bincount(encoded_indices.flatten())
    top_features = np.argsort(feature_counts)[-n_features:]
    
    for feature_idx in top_features:
        activations = []
        hover_texts = []
        
        for step, (acts, text) in enumerate(zip(prepared_acts, generated_texts)):
            indices = acts['encoded']['top_indices'][0, -1]
            values = acts['encoded']['top_acts'][0, -1]
            
            mask = indices == feature_idx
            activation = float(values[mask][0]) if mask.any() else 0.0
            activations.append(activation)
            
            token_text = text[-1] if step > 0 else ""
            hover_text = f"Step {step}<br>Feature: {feature_idx}<br>Activation: {activation:.3f}"
            if token_text:
                hover_text += f"<br>Token: '{token_text}'"
            hover_texts.append(hover_text)
            
        fig.add_trace(go.Scatter(
            y=activations,
            name=f"Feature {feature_idx}",
            mode='lines+markers',
            hovertext=hover_texts,
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title_text=title,
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

class ExperimentResultsVisualizer(BaseVisualizer):
    """Visualizer for experiment results."""
    def __init__(self, results: Union[ExperimentResults, Dict], num_runs: int = 5):
        super().__init__(results)
        self.num_runs = num_runs
    
    def create_figures(self) -> List[go.Figure]:
        """Create experiment results visualization figures."""
        return [
            self._create_stopping_reasons_plot(),
            self._create_token_frequency_plot(),
            self._create_statistics_plot(),
            self._create_samples_table()
        ]
    
    def _create_stopping_reasons_plot(self) -> go.Figure:
        """Creates pie chart of stopping reasons."""
        labels = list(self.results['stopping_reasons'].keys())
        values = list(self.results['stopping_reasons'].values())
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent')])
        fig.update_layout(
            title=f"{self.base_title}<br>Early Stopping Reasons",
            height=600, width=800
        )
        return fig
    
    def _create_token_frequency_plot(self) -> go.Figure:
        """Creates bar chart of most common tokens."""
        # Get top 10 tokens
        tokens = list(self.results['token_frequencies'].keys())[:10]
        counts = list(self.results['token_frequencies'].values())[:10]
        
        fig = go.Figure(data=[go.Bar(x=tokens, y=counts)])
        fig.update_layout(
            title=f"{self.base_title}<br>Most Common Tokens",
            xaxis_title="Token",
            yaxis_title="Frequency",
            height=600, width=800
        )
        return fig
    
    def _create_statistics_plot(self) -> go.Figure:
        """Creates statistics indicators showing averages across runs."""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Add average length indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=self.results['avg_length'],
                title=f"Average Length<br>({self.num_runs} runs)",
            ),
            row=1, col=1
        )
        
        # Add unique token ratio indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=self.results['unique_ratio'] * 100,
                title=f"Unique Token %<br>({self.num_runs} runs)",
                gauge={'axis': {'range': [0, 100]}}
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"{self.base_title}<br>Generation Statistics",
            height=400, width=800
        )
        return fig
    
    def _create_samples_table(self) -> go.Figure:
        """Creates table showing sample generations and their stopping reasons."""
        # Show samples up to num_runs
        samples = self.results['all_texts'][:self.num_runs]
        
        # Get the stopping reasons from the Counter
        stopping_reasons = self.results['stopping_reasons']
        reasons = []
        for i in range(len(samples)):
            # For each sample, find the first stopping reason that has a count > 0
            for reason, count in stopping_reasons.items():
                if count > 0:
                    reasons.append(reason)
                    stopping_reasons[reason] -= 1  # Decrement count after using
                    break
            if len(reasons) < i + 1:  # If no reason found
                reasons.append("Unknown")  # Shouldn't happen but just in case
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[
                    f'Sample Generations (showing {len(samples)} of {self.num_runs})',
                    'Stopping Reason'
                ],
                font=dict(size=12),
                align="left"
            ),
            cells=dict(
                values=[samples, reasons],
                font=dict(size=11),
                align="left",
                height=30
            )
        )])
        
        fig.update_layout(
            title=f"{self.base_title}<br>Sample Generations",
            height=400, width=1000  # Made wider to accommodate two columns
        )
        return fig

def prepare_for_visualization(generation_acts: List) -> List[Dict]:
    """Convert GPU tensors to numpy arrays for visualization."""
    prepared_acts = []
    for acts in generation_acts:
        prepared = {
            'original': acts[0]['original'].cpu().numpy(),
            'encoded': {
                'top_acts': acts[0]['encoded'].top_acts.cpu().numpy(),
                'top_indices': acts[0]['encoded'].top_indices.cpu().numpy()
            }
        }
        prepared_acts.append(prepared)
    return prepared_acts

def visualize_generation_activations(generation_acts, generated_texts, title_prefix=""):
    """Single entry point for visualization with clear data conversion."""
    title_prefix = f"{title_prefix} - " if title_prefix else ""
    prepared_acts = prepare_for_visualization(generation_acts)
    
    return [
        create_activation_heatmaps(
            prepared_acts, 
            generated_texts,
            title=f"{title_prefix}Feature Activations During Generation (MLP Layer 10)"
        ),
        create_feature_evolution_plot(
            prepared_acts=prepared_acts,
            generated_texts=generated_texts,
            n_features=10,
            title=f"{title_prefix}Top Feature Evolution During Generation (MLP Layer 10)"
        )
    ]

def visualize_experiment_results(
    results: Union[ExperimentResults, Dict],
    num_runs: int = 5
) -> List[go.Figure]:
    return ExperimentResultsVisualizer(results, num_runs).create_figures()

def collect_activations(outputs, sae) -> List[Dict[str, torch.Tensor]]:
    """Keep data on GPU until computation is complete."""
    step_acts = []
    # Stay on GPU for potential further computations
    hidden_state = outputs.hidden_states[10].detach()
    encoded = sae.encode(hidden_state)
    
    step_acts.append({
        'original': hidden_state,
        'encoded': encoded
    })
    return step_acts

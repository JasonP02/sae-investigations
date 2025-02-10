from typing import List, Tuple, Dict, Union
import numpy as np
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

class GenerationActivationVisualizer(BaseVisualizer):
    """Visualizer for generation activations."""
    def create_figures(self) -> List[go.Figure]:
        """Create activation visualization figures."""
        figures = []
        
        if 'generation_acts' not in self.results or 'generated_texts' not in self.results:
            return figures
        
        # Extract data
        generation_acts = self.results['generation_acts']
        generated_texts = self.results['generated_texts']
        layer_idx = self.results.get('layer_idx', 0)
        step_interval = self.results.get('step_interval', 1)
        
        # 1. Feature activation heatmap
        fig_heatmap = self._create_activation_heatmap(
            generation_acts, generated_texts, layer_idx, step_interval
        )
        figures.append(self._validate_figure(fig_heatmap))
        
        # 2. Feature usage evolution
        fig_evolution = self._create_feature_evolution_plot(
            generation_acts, generated_texts, layer_idx, step_interval
        )
        figures.append(self._validate_figure(fig_evolution))
        
        return figures
    
    def _create_activation_heatmap(
        self,
        generation_acts: List,
        generated_texts: List[str],
        layer_idx: int,
        step_interval: int
    ) -> go.Figure:
        """Creates a heatmap of feature activations over generation steps."""
        # Create subplots for original and encoded activations
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Original MLP Activations', 'SAE-Encoded Activations'),
            vertical_spacing=0.15
        )
        
        # Get activations for specified layer across all steps
        step_acts_original = []
        step_acts_encoded = []
        step_indices = []
        step_labels = []
        
        for step, (acts, text) in enumerate(zip(generation_acts, generated_texts)):
            if step % step_interval == 0:
                layer_acts = acts[0]
                # Get original activations
                step_acts_original.append(layer_acts['original'].cpu().numpy()[0, -1])
                # Get encoded activations and indices
                encoded_acts = layer_acts['encoded']
                step_acts_encoded.append(encoded_acts.top_acts.detach().cpu().numpy()[0, -1])
                step_indices.append(encoded_acts.top_indices.detach().cpu().numpy()[0, -1])
                step_labels.append(f"Step {step}: {text[-20:]}")
        
        # Convert to numpy arrays
        step_acts_original = np.array(step_acts_original)
        step_acts_encoded = np.array(step_acts_encoded)
        step_indices = np.array(step_indices)
        
        # Create heatmap for original activations
        fig.add_trace(
            go.Heatmap(
                z=step_acts_original,
                y=step_labels,
                colorscale='RdBu',
                name='Original Values',
                showscale=True,
                colorbar=dict(title="Activation Value", y=0.85, len=0.35)
            ),
            row=1, col=1
        )
        
        # Create heatmap for encoded activations
        fig.add_trace(
            go.Heatmap(
                z=step_acts_encoded,
                y=step_labels,
                colorscale='RdBu',
                name='Encoded Values',
                showscale=True,
                colorbar=dict(title="Activation Value", y=0.35, len=0.35)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title_text=f"{self.base_title}<br>Feature Activations During Generation (MLP Layer 10)",
            height=1200,  # Made taller to accommodate both plots
            width=1200
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Hidden Dimension", row=1, col=1)
        fig.update_xaxes(title_text="K-index", row=2, col=1)
        fig.update_yaxes(title_text="Generation Step", row=1, col=1)
        fig.update_yaxes(title_text="Generation Step", row=2, col=1)
        
        return fig
    
    def _create_feature_evolution_plot(
        self,
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
                encoded_acts = layer_acts['encoded']
                step_acts.append(encoded_acts.top_acts.detach().cpu().numpy()[0, -1])
                step_indices.append(encoded_acts.top_indices.detach().cpu().numpy()[0, -1])
        
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
            title_text=f"{self.base_title}<br>Top {N} Feature Evolution During Generation (MLP Layer 10)",
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

# Simple wrapper functions for backward compatibility
def visualize_generation_activations(
    generation_acts: List,
    generated_texts: List[str],
    layer_idx: int = 0,
    step_interval: int = 1
) -> List[go.Figure]:
    results = {
        'generation_acts': generation_acts,
        'generated_texts': generated_texts,
        'layer_idx': layer_idx,
        'step_interval': step_interval
    }
    return GenerationActivationVisualizer(results).create_figures()

def visualize_experiment_results(
    results: Union[ExperimentResults, Dict],
    num_runs: int = 5
) -> List[go.Figure]:
    return ExperimentResultsVisualizer(results, num_runs).create_figures()
"""
Visualization utilities for nucleotide biology analysis results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SAEVisualizationMixin:
    """
    Visualization methods for Sparse Autoencoder analysis.
    """

    @staticmethod
    def plot_training_metrics(metrics, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot SAE training metrics.

        Args:
            metrics: SAETrainingMetrics object
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot losses
        epochs = range(len(metrics.total_loss))
        axes[0].plot(epochs, metrics.reconstruction_loss, label='Reconstruction', alpha=0.8)
        axes[0].plot(epochs, metrics.sparsity_loss, label='Sparsity', alpha=0.8)
        axes[0].plot(epochs, metrics.total_loss, label='Total', alpha=0.8, linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot feature density
        axes[1].plot(epochs, metrics.feature_density, color='green', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Average Active Features')
        axes[1].set_title('Feature Sparsity')
        axes[1].grid(True, alpha=0.3)

        # Plot explained variance
        axes[2].plot(epochs, metrics.explained_variance, color='red', alpha=0.8)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Explained Variance (R²)')
        axes[2].set_title('Reconstruction Quality')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_activations(features: torch.Tensor, 
                                top_k: int = 50,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot feature activation patterns.

        Args:
            features: Feature activation tensor [n_samples, n_features]
            top_k: Number of top features to show
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        # Get top active features
        mean_activations = np.mean(features, axis=0)
        top_indices = np.argsort(mean_activations)[-top_k:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Heatmap of top features
        top_features = features[:, top_indices]
        sns.heatmap(top_features.T, ax=ax1, cmap='viridis', 
                   cbar_kws={'label': 'Activation'})
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Feature Index')
        ax1.set_title(f'Top {top_k} Feature Activations')

        # Distribution of mean activations
        ax2.hist(mean_activations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(mean_activations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(mean_activations):.4f}')
        ax2.set_xlabel('Mean Activation')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Distribution of Feature Activations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_interpretation_summary(interpretations: List, 
                                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot summary of feature interpretations.

        Args:
            interpretations: List of FeatureInterpretation objects
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not interpretations:
            logger.warning("No interpretations to plot")
            return plt.figure(figsize=figsize)

        # Extract data
        confidence_scores = [interp.confidence_score for interp in interpretations]
        n_sequences = [len(interp.top_activating_sequences) for interp in interpretations]
        functions = [interp.biological_function or 'unknown' for interp in interpretations]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Confidence score distribution
        axes[0, 0].hist(confidence_scores, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Feature Interpretation Confidence')
        axes[0, 0].grid(True, alpha=0.3)

        # Number of activating sequences
        axes[0, 1].hist(n_sequences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Number of Activating Sequences')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Feature Specificity')
        axes[0, 1].grid(True, alpha=0.3)

        # Biological function distribution
        function_counts = pd.Series(functions).value_counts()
        function_counts.plot(kind='bar', ax=axes[1, 0], color='salmon')
        axes[1, 0].set_xlabel('Biological Function')
        axes[1, 0].set_ylabel('Number of Features')
        axes[1, 0].set_title('Predicted Biological Functions')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Confidence vs specificity scatter
        axes[1, 1].scatter(n_sequences, confidence_scores, alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Number of Activating Sequences')
        axes[1, 1].set_ylabel('Confidence Score')
        axes[1, 1].set_title('Confidence vs Specificity')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

class AttributionVisualizationMixin:
    """
    Visualization methods for attribution graph analysis.
    """

    @staticmethod
    def plot_attribution_graph(attribution_graph, 
                              layout: str = 'spring',
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot attribution graph using networkx.

        Args:
            attribution_graph: AttributionGraph object
            layout: Layout algorithm ('spring', 'circular', 'hierarchical')
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not attribution_graph.nodes:
            logger.warning("No nodes in attribution graph")
            return plt.figure(figsize=figsize)

        fig, ax = plt.subplots(figsize=figsize)

        G = attribution_graph.graph

        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'hierarchical':
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if hasattr(nx, 'nx_agraph') else nx.spring_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Extract node attributes
        node_sizes = []
        node_colors = []

        for node_id in G.nodes():
            if node_id in attribution_graph.nodes:
                node = attribution_graph.nodes[node_id]
                node_sizes.append(max(100, abs(node.attribution_score) * 1000))
                node_colors.append(node.layer)
            else:
                node_sizes.append(100)
                node_colors.append(0)

        # Extract edge weights
        edge_weights = []
        for edge in G.edges():
            if edge in attribution_graph.edges:
                edge_weights.append(abs(attribution_graph.edges[edge].weight))
            else:
                edge_weights.append(1.0)

        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                              cmap='viridis', alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in edge_weights], 
                              alpha=0.6, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        ax.set_title('Attribution Graph')
        ax.axis('off')

        # Add colorbar for layers
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Layer')

        return fig

    @staticmethod
    def plot_feature_pathway(pathway: List[Tuple[int, int]], 
                           pathway_name: str = "Feature Pathway",
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot a specific feature pathway.

        Args:
            pathway: List of (layer, feature) tuples
            pathway_name: Name of the pathway
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        layers = [step[0] for step in pathway]
        features = [step[1] for step in pathway]

        # Create pathway plot
        ax.plot(layers, features, 'o-', linewidth=2, markersize=8, alpha=0.8)

        # Add labels
        for i, (layer, feature) in enumerate(pathway):
            ax.annotate(f'L{layer}F{feature}', (layer, feature), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Feature Index')
        ax.set_title(f'{pathway_name}')
        ax.grid(True, alpha=0.3)

        return fig

class InterventionVisualizationMixin:
    """
    Visualization methods for intervention experiments.
    """

    @staticmethod
    def plot_intervention_effects(intervention_results: List, 
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot results from intervention experiments.

        Args:
            intervention_results: List of InterventionResult objects
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not intervention_results:
            logger.warning("No intervention results to plot")
            return plt.figure(figsize=figsize)

        # Extract data
        intervention_types = [result.intervention_type for result in intervention_results]
        effect_sizes = [result.effect_size for result in intervention_results]
        layers = [result.target_layer for result in intervention_results]
        strengths = [result.intervention_strength for result in intervention_results]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Effect sizes by intervention type
        df = pd.DataFrame({
            'intervention_type': intervention_types,
            'effect_size': effect_sizes,
            'layer': layers,
            'strength': strengths
        })

        sns.boxplot(data=df, x='intervention_type', y='effect_size', ax=axes[0, 0])
        axes[0, 0].set_title('Effect Sizes by Intervention Type')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Effect sizes by layer
        sns.scatterplot(data=df, x='layer', y='effect_size', 
                       hue='intervention_type', ax=axes[0, 1])
        axes[0, 1].set_title('Effect Sizes by Layer')

        # Intervention strength vs effect size
        sns.scatterplot(data=df, x='strength', y='effect_size', 
                       hue='intervention_type', ax=axes[1, 0])
        axes[1, 0].set_title('Intervention Strength vs Effect Size')

        # Distribution of effect sizes
        axes[1, 1].hist(effect_sizes, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 1].set_xlabel('Effect Size')
        axes[1, 1].set_ylabel('Number of Interventions')
        axes[1, 1].set_title('Distribution of Effect Sizes')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_causal_validation(validation_results: Dict[str, Any],
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot causal claim validation results.

        Args:
            validation_results: Results from validate_causal_claim
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Effect sizes across trials
        effect_sizes = validation_results.get('effect_sizes', [])
        trial_numbers = range(1, len(effect_sizes) + 1)

        ax1.plot(trial_numbers, effect_sizes, 'o-', alpha=0.8)
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Effect Size')
        ax1.set_title('Effect Sizes Across Trials')
        ax1.grid(True, alpha=0.3)

        if effect_sizes:
            ax1.axhline(np.mean(effect_sizes), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(effect_sizes):.4f}')
            ax1.legend()

        # Validation summary
        consistency = validation_results.get('consistency_score', 0)
        success = validation_results.get('validation_success', False)

        categories = ['Consistency Score']
        values = [consistency]
        colors = ['green' if success else 'red']

        bars = ax2.bar(categories, values, color=colors, alpha=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Score')
        ax2.set_title(f'Validation {"Success" if success else "Failure"}')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        return fig

class ComprehensiveVisualizer:
    """
    Main visualization class that combines all visualization capabilities.
    """

    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.default_figsize = figsize

    def create_analysis_dashboard(self, 
                                results: Dict[str, Any],
                                save_path: Optional[str] = None) -> List[plt.Figure]:
        """
        Create comprehensive analysis dashboard.

        Args:
            results: Complete analysis results
            save_path: Optional path to save figures

        Returns:
            List of matplotlib figures
        """
        figures = []

        # SAE training metrics (if available)
        if 'sae_metrics' in results:
            for layer, metrics in results['sae_metrics'].items():
                fig = SAEVisualizationMixin.plot_training_metrics(metrics)
                fig.suptitle(f'SAE Training Metrics - Layer {layer}')
                figures.append(fig)

        # Feature interpretations (if available)
        if 'interpretations' in results:
            for layer, interpretations in results['interpretations'].items():
                if interpretations:
                    fig = SAEVisualizationMixin.plot_feature_interpretation_summary(interpretations)
                    fig.suptitle(f'Feature Interpretations - Layer {layer}')
                    figures.append(fig)

        # Attribution graphs (if available)
        if 'attribution_graphs' in results:
            for seq_id, graph in results['attribution_graphs'].items():
                fig = AttributionVisualizationMixin.plot_attribution_graph(graph)
                fig.suptitle(f'Attribution Graph - {seq_id}')
                figures.append(fig)

        # Intervention results (if available)
        if 'intervention_results' in results:
            fig = InterventionVisualizationMixin.plot_intervention_effects(
                results['intervention_results']
            )
            fig.suptitle('Intervention Experiment Results')
            figures.append(fig)

        # Save figures if requested
        if save_path:
            for i, fig in enumerate(figures):
                fig.savefig(f'{save_path}_figure_{i}.png', dpi=300, bbox_inches='tight')

        return figures

    def create_interactive_dashboard(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create interactive dashboard using Plotly.

        Args:
            results: Analysis results

        Returns:
            Plotly figure
        """
        # This would create an interactive dashboard
        # For now, return a simple placeholder
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], 
                               name="Placeholder", mode='lines+markers'))
        fig.update_layout(title="Interactive Dashboard (Placeholder)")

        return fig

# Utility functions
def save_all_figures(figures: List[plt.Figure], 
                    output_dir: str, 
                    prefix: str = "analysis") -> List[str]:
    """
    Save all figures to files.

    Args:
        figures: List of matplotlib figures
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    saved_paths = []

    for i, fig in enumerate(figures):
        filename = f"{prefix}_figure_{i:03d}.png"
        filepath = f"{output_dir}/{filename}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        saved_paths.append(filepath)
        logger.info(f"Saved figure to {filepath}")

    return saved_paths

def create_summary_plot(results: Dict[str, Any], 
                       figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create a summary plot of all analysis results.

    Args:
        results: Complete analysis results
        figsize: Figure size

    Returns:
        Summary figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Nucleotide Biology Analysis Summary', fontsize=16)

    # Placeholder plots - would be filled with actual summary data
    for i, ax in enumerate(axes.flat):
        ax.text(0.5, 0.5, f'Summary Plot {i+1}\n(To be implemented)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Analysis Component {i+1}')

    plt.tight_layout()
    return fig

"""
Attribution Graphs implementation for circuit tracing in Nucleotide Transformer
Based on Anthropic's "On the Biology of a Large Language Model" methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass
import networkx as nx
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttributionNode:
    """
    Represents a node in the attribution graph.
    """

    feature_idx: int
    layer: int
    activation: float
    attribution_score: float
    interpretation: Optional[str] = None
    examples: Optional[List[str]] = None


@dataclass
class AttributionEdge:
    """
    Represents an edge in the attribution graph.
    """

    source_node: AttributionNode
    target_node: AttributionNode
    weight: float
    attribution_method: str


class ReplacementModel(nn.Module):
    """
    Replacement model that uses interpretable SAE features instead of raw neurons.
    This is the core component that enables circuit tracing.
    """

    def __init__(
        self, original_model, sae_models: Dict[int, nn.Module], device: str = "cuda"
    ):
        """
        Initialize replacement model.

        Args:
            original_model: Original nucleotide transformer
            sae_models: Dictionary mapping layer index to trained SAE
            device: Device to run on
        """
        super().__init__()
        self.original_model = original_model
        self.sae_models = sae_models
        self.device = device
        self.layer_features = {}  # Store features for each layer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass through replacement model.

        Args:
            input_ids: Tokenized sequences
            attention_mask: Attention mask
            return_features: Whether to return SAE features

        Returns:
            Dictionary containing outputs and features
        """
        # Get original model outputs with all hidden states
        with torch.no_grad():
            original_outputs = self.original_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

        results = {
            "logits": original_outputs.logits,
            "hidden_states": original_outputs.hidden_states,
            "attentions": original_outputs.attentions,
        }

        if return_features:
            layer_features = {}

            # Extract SAE features for each layer
            for layer_idx, sae in self.sae_models.items():
                if layer_idx < len(original_outputs.hidden_states):
                    hidden_state = original_outputs.hidden_states[layer_idx]

                    # Pool over sequence dimension (mean pooling excluding padding)
                    mask = attention_mask.unsqueeze(-1).float()
                    pooled_hidden = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

                    # Get SAE features
                    with torch.no_grad():
                        reconstructed, features = sae(pooled_hidden)

                    layer_features[layer_idx] = {
                        "features": features,
                        "reconstructed": reconstructed,
                        "original": pooled_hidden,
                    }

            results["layer_features"] = layer_features
            self.layer_features = layer_features

        return results


class AttributionAnalyzer:
    """
    Analyzes causal relationships between features using attribution methods.
    """

    def __init__(
        self,
        replacement_model: ReplacementModel,
        attribution_method: str = "integrated_gradients",
        n_steps: int = 50,
    ):
        """
        Initialize attribution analyzer.

        Args:
            replacement_model: The replacement model with SAE features
            attribution_method: Method for computing attributions
            n_steps: Number of steps for gradient-based methods
        """
        self.replacement_model = replacement_model
        self.attribution_method = attribution_method
        self.n_steps = n_steps

    def compute_feature_attributions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_layer: int,
        target_feature: int,
        baseline_method: str = "zero",
    ) -> Dict[int, torch.Tensor]:
        """
        Compute attributions from all features to a target feature.

        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            target_layer: Layer of target feature
            target_feature: Index of target feature
            baseline_method: Method for creating baseline

        Returns:
            Dictionary mapping source layer to attribution scores
        """
        if self.attribution_method == "integrated_gradients":
            return self._integrated_gradients_attribution(
                input_ids, attention_mask, target_layer, target_feature, baseline_method
            )
        elif self.attribution_method == "gradient_shap":
            return self._gradient_shap_attribution(
                input_ids, attention_mask, target_layer, target_feature
            )
        else:
            raise ValueError(f"Unknown attribution method: {self.attribution_method}")

    def _integrated_gradients_attribution(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_layer: int,
        target_feature: int,
        baseline_method: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute integrated gradients attribution.
        """
        # Create baseline
        if baseline_method == "zero":
            baseline_ids = torch.zeros_like(input_ids)
        elif baseline_method == "random":
            baseline_ids = torch.randint_like(
                input_ids, 0, self.replacement_model.original_model.config.vocab_size
            )
        else:
            raise ValueError(f"Unknown baseline method: {baseline_method}")

        attributions = {}

        for step in range(self.n_steps):
            alpha = step / (self.n_steps - 1) if self.n_steps > 1 else 1.0

            # Interpolate between baseline and input
            interpolated_ids = baseline_ids + alpha * (input_ids - baseline_ids)
            interpolated_ids = interpolated_ids.long()

            # Require gradients for interpolated input
            interpolated_ids.requires_grad_(True)

            # Forward pass
            outputs = self.replacement_model(interpolated_ids, attention_mask)

            # Get target feature activation
            target_activation = outputs["layer_features"][target_layer]["features"][
                0, target_feature
            ]

            # Compute gradients
            gradients = torch.autograd.grad(
                target_activation,
                interpolated_ids,
                retain_graph=True,
                create_graph=False,
            )[0]

            # Accumulate gradients
            if step == 0:
                integrated_gradients = gradients
            else:
                integrated_gradients += gradients

        # Average and multiply by input difference
        integrated_gradients = integrated_gradients / self.n_steps
        integrated_gradients *= (input_ids - baseline_ids).float()

        # Convert token-level gradients to feature-level attributions
        for layer_idx in outputs["layer_features"].keys():
            # For simplicity, we'll use the sum of token gradients as layer attribution
            # In a full implementation, this would involve backpropagating through SAE
            layer_attribution = integrated_gradients.sum(
                dim=1
            )  # Sum over sequence dimension
            attributions[layer_idx] = layer_attribution

        return attributions

    def _gradient_shap_attribution(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_layer: int,
        target_feature: int,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute GradientSHAP attribution.
        """
        # Simplified implementation - would need proper baseline sampling
        return self._integrated_gradients_attribution(
            input_ids, attention_mask, target_layer, target_feature, "random"
        )


class AttributionGraph:
    """
    Represents the attribution graph showing causal relationships between features.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.edges = {}

    def add_node(self, node: AttributionNode):
        """Add a node to the attribution graph."""
        node_id = f"L{node.layer}_F{node.feature_idx}"
        self.graph.add_node(node_id)
        self.nodes[node_id] = node

    def add_edge(self, edge: AttributionEdge):
        """Add an edge to the attribution graph."""
        source_id = f"L{edge.source_node.layer}_F{edge.source_node.feature_idx}"
        target_id = f"L{edge.target_node.layer}_F{edge.target_node.feature_idx}"

        self.graph.add_edge(source_id, target_id, weight=edge.weight)
        self.edges[(source_id, target_id)] = edge

    def prune_graph(
        self, min_attribution: float = 0.01, max_nodes: int = 50
    ) -> "AttributionGraph":
        """
        Prune the graph to keep only the most important nodes and edges.

        Args:
            min_attribution: Minimum attribution score to keep
            max_nodes: Maximum number of nodes to keep

        Returns:
            Pruned attribution graph
        """
        pruned_graph = AttributionGraph()

        # Sort nodes by attribution score
        sorted_nodes = sorted(
            self.nodes.items(), key=lambda x: abs(x[1].attribution_score), reverse=True
        )

        # Keep top nodes
        for node_id, node in sorted_nodes[:max_nodes]:
            if abs(node.attribution_score) >= min_attribution:
                pruned_graph.add_node(node)

        # Keep edges between retained nodes
        for (source_id, target_id), edge in self.edges.items():
            if (
                source_id in pruned_graph.nodes
                and target_id in pruned_graph.nodes
                and abs(edge.weight) >= min_attribution
            ):
                pruned_graph.add_edge(edge)

        return pruned_graph

    def get_feature_pathways(
        self, target_feature: str, max_depth: int = 3
    ) -> List[List[str]]:
        """
        Get all pathways leading to a target feature.

        Args:
            target_feature: Target feature node ID
            max_depth: Maximum pathway depth

        Returns:
            List of pathways (each pathway is a list of node IDs)
        """
        pathways = []

        def dfs_pathways(current_node, path, depth):
            if depth >= max_depth:
                return

            path = path + [current_node]

            # Get predecessors (nodes that influence current node)
            predecessors = list(self.graph.predecessors(current_node))

            if not predecessors:
                # This is a leaf node (start of pathway)
                pathways.append(list(reversed(path)))
            else:
                for pred in predecessors:
                    dfs_pathways(pred, path, depth + 1)

        dfs_pathways(target_feature, [], 0)
        return pathways


class CircuitTracer:
    """
    Main class for performing circuit tracing analysis on Nucleotide Transformer.
    """

    def __init__(
        self, nucleotide_model, sae_models: Dict[int, nn.Module], device: str = "cuda"
    ):
        """
        Initialize circuit tracer.

        Args:
            nucleotide_model: Nucleotide transformer model
            sae_models: Dictionary of trained SAE models for each layer
            device: Device to run on
        """
        self.nucleotide_model = nucleotide_model
        self.replacement_model = ReplacementModel(nucleotide_model, sae_models, device)
        self.attribution_analyzer = AttributionAnalyzer(self.replacement_model)
        self.device = device

    def trace_sequence_processing(
        self,
        sequences: List[str],
        target_layer: int,
        target_feature: int,
        feature_threshold: float = 0.01,
    ) -> AttributionGraph:
        """
        Trace how a sequence is processed through the network.

        Args:
            sequences: DNA sequences to analyze
            target_layer: Layer containing target feature
            target_feature: Feature to trace back from
            feature_threshold: Minimum activation to consider

        Returns:
            Attribution graph showing processing pathway
        """
        logger.info(f"Tracing sequence processing for {len(sequences)} sequences")

        # Tokenize sequences
        tokenized = self.nucleotide_model.tokenize_sequences(sequences)

        # Get model outputs and features
        outputs = self.replacement_model(
            tokenized["input_ids"], tokenized["attention_mask"]
        )

        # Build attribution graph
        attribution_graph = AttributionGraph()

        # Add nodes for all active features
        for layer_idx, layer_data in outputs["layer_features"].items():
            features = layer_data["features"][0]  # Take first sequence for now

            for feature_idx, activation in enumerate(features):
                if activation.item() > feature_threshold:
                    node = AttributionNode(
                        feature_idx=feature_idx,
                        layer=layer_idx,
                        activation=activation.item(),
                        attribution_score=activation.item(),  # Will be updated below
                    )
                    attribution_graph.add_node(node)

        # Compute attributions between layers
        attributions = self.attribution_analyzer.compute_feature_attributions(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            target_layer,
            target_feature,
        )

        # Add edges based on attributions
        for source_layer, layer_attributions in attributions.items():
            if source_layer in outputs["layer_features"]:
                source_features = outputs["layer_features"][source_layer]["features"][0]

                for source_feature_idx, source_activation in enumerate(source_features):
                    if source_activation.item() > feature_threshold:
                        source_node_id = f"L{source_layer}_F{source_feature_idx}"
                        target_node_id = f"L{target_layer}_F{target_feature}"

                        if (
                            source_node_id in attribution_graph.nodes
                            and target_node_id in attribution_graph.nodes
                        ):

                            # Create edge
                            edge = AttributionEdge(
                                source_node=attribution_graph.nodes[source_node_id],
                                target_node=attribution_graph.nodes[target_node_id],
                                weight=layer_attributions[0].item(),  # Simplified
                                attribution_method=self.attribution_analyzer.attribution_method,
                            )
                            attribution_graph.add_edge(edge)

        return attribution_graph.prune_graph()

    def analyze_motif_detection(
        self,
        sequences_with_motif: List[str],
        sequences_without_motif: List[str],
        motif_name: str,
    ) -> Dict[str, Any]:
        """
        Analyze how the model detects specific DNA motifs.

        Args:
            sequences_with_motif: Sequences containing the motif
            sequences_without_motif: Sequences without the motif
            motif_name: Name of the motif being analyzed

        Returns:
            Analysis results
        """
        logger.info(f"Analyzing motif detection for {motif_name}")

        results = {
            "motif_name": motif_name,
            "differential_features": {},
            "motif_specific_circuits": {},
        }

        # Get features for both sequence types
        with_motif_features = self._get_sequence_features(sequences_with_motif)
        without_motif_features = self._get_sequence_features(sequences_without_motif)

        # Find differentially active features
        for layer_idx in with_motif_features.keys():
            if layer_idx in without_motif_features:
                with_features = with_motif_features[layer_idx]
                without_features = without_motif_features[layer_idx]

                # Compute mean activations
                with_mean = with_features.mean(dim=0)
                without_mean = without_features.mean(dim=0)

                # Find features more active in motif sequences
                differential = with_mean - without_mean
                motif_features = torch.where(differential > 0.1)[0]

                results["differential_features"][layer_idx] = {
                    "feature_indices": motif_features.tolist(),
                    "differential_scores": differential[motif_features].tolist(),
                }

        return results

    def _get_sequence_features(self, sequences: List[str]) -> Dict[int, torch.Tensor]:
        """Extract features for a list of sequences."""
        all_features = defaultdict(list)

        for seq in sequences:
            tokenized = self.nucleotide_model.tokenize_sequences([seq])
            outputs = self.replacement_model(
                tokenized["input_ids"], tokenized["attention_mask"]
            )

            for layer_idx, layer_data in outputs["layer_features"].items():
                all_features[layer_idx].append(layer_data["features"])

        # Stack features
        return {
            layer_idx: torch.cat(features_list, dim=0)
            for layer_idx, features_list in all_features.items()
        }

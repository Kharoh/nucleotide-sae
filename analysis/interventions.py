"""
Interventions module for validating circuit hypotheses through targeted feature manipulations
Based on Anthropic's intervention experiments in "On the Biology of a Large Language Model"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
from dataclasses import dataclass
import logging
from collections import defaultdict
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InterventionResult:
    """
    Results from a feature intervention experiment.
    """
    intervention_type: str
    target_layer: int
    target_features: List[int]
    intervention_strength: float
    original_output: Any
    modified_output: Any
    effect_size: float
    downstream_effects: Dict[str, float]

@dataclass
class CausalClaim:
    """
    Represents a causal claim to be tested through interventions.
    """
    source_layer: int
    source_features: List[int]
    target_layer: int
    target_features: List[int]
    claim_description: str
    expected_effect_direction: str  # "positive", "negative", "bidirectional"

class FeatureInterventioner:
    """
    Performs targeted interventions on SAE features to test causal hypotheses.
    """

    def __init__(self, 
                 nucleotide_model,
                 sae_models: Dict[int, nn.Module],
                 device: str = "cuda"):
        """
        Initialize feature interventioner.

        Args:
            nucleotide_model: Nucleotide transformer model
            sae_models: Dictionary of trained SAE models
            device: Device to run on
        """
        self.nucleotide_model = nucleotide_model
        self.sae_models = sae_models
        self.device = device

    def suppress_features(self, 
                         layer: int,
                         feature_indices: List[int],
                         suppression_factor: float = 0.0) -> Callable:
        """
        Create a hook to suppress specific features.

        Args:
            layer: Layer to intervene on
            feature_indices: Indices of features to suppress
            suppression_factor: Factor to multiply features by (0.0 = complete suppression)

        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            if hasattr(module, 'encode'):  # This is an SAE module
                # Get the features
                features = module.encode(input[0])

                # Suppress specified features
                modified_features = features.clone()
                modified_features[:, feature_indices] *= suppression_factor

                # Decode back
                modified_output = module.decode(modified_features)
                return modified_output

            return output

        return hook_fn

    def amplify_features(self, 
                        layer: int,
                        feature_indices: List[int],
                        amplification_factor: float = 2.0) -> Callable:
        """
        Create a hook to amplify specific features.

        Args:
            layer: Layer to intervene on
            feature_indices: Indices of features to amplify
            amplification_factor: Factor to multiply features by

        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            if hasattr(module, 'encode'):
                features = module.encode(input[0])

                # Amplify specified features
                modified_features = features.clone()
                modified_features[:, feature_indices] *= amplification_factor

                modified_output = module.decode(modified_features)
                return modified_output

            return output

        return hook_fn

    def inject_features(self, 
                       layer: int,
                       feature_values: Dict[int, float]) -> Callable:
        """
        Create a hook to inject specific feature values.

        Args:
            layer: Layer to intervene on
            feature_values: Dictionary mapping feature index to desired value

        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            if hasattr(module, 'encode'):
                features = module.encode(input[0])

                # Inject specified feature values
                modified_features = features.clone()
                for feature_idx, value in feature_values.items():
                    modified_features[:, feature_idx] = value

                modified_output = module.decode(modified_features)
                return modified_output

            return output

        return hook_fn

    def test_feature_suppression(self, 
                               sequences: List[str],
                               target_layer: int,
                               target_features: List[int],
                               suppression_strength: float = 0.0,
                               measure_downstream: bool = True) -> InterventionResult:
        """
        Test the effect of suppressing specific features.

        Args:
            sequences: DNA sequences to test
            target_layer: Layer containing features to suppress
            target_features: Features to suppress
            suppression_strength: Suppression strength (0.0 = complete suppression)
            measure_downstream: Whether to measure effects on downstream layers

        Returns:
            Intervention results
        """
        logger.info(f"Testing suppression of features {target_features} in layer {target_layer}")

        # Get original outputs
        original_outputs = self._get_model_outputs(sequences)

        # Apply intervention
        if target_layer in self.sae_models:
            hook = self.suppress_features(target_layer, target_features, suppression_strength)
            handle = self.sae_models[target_layer].register_forward_hook(hook)

            try:
                modified_outputs = self._get_model_outputs(sequences)
            finally:
                handle.remove()
        else:
            raise ValueError(f"No SAE model available for layer {target_layer}")

        # Compute effect size
        effect_size = self._compute_output_difference(original_outputs, modified_outputs)

        # Measure downstream effects if requested
        downstream_effects = {}
        if measure_downstream:
            downstream_effects = self._measure_downstream_effects(
                original_outputs, modified_outputs, target_layer
            )

        return InterventionResult(
            intervention_type="suppression",
            target_layer=target_layer,
            target_features=target_features,
            intervention_strength=suppression_strength,
            original_output=original_outputs,
            modified_output=modified_outputs,
            effect_size=effect_size,
            downstream_effects=downstream_effects
        )

    def test_feature_amplification(self, 
                                 sequences: List[str],
                                 target_layer: int,
                                 target_features: List[int],
                                 amplification_strength: float = 2.0) -> InterventionResult:
        """
        Test the effect of amplifying specific features.

        Args:
            sequences: DNA sequences to test
            target_layer: Layer containing features to amplify
            target_features: Features to amplify
            amplification_strength: Amplification factor

        Returns:
            Intervention results
        """
        logger.info(f"Testing amplification of features {target_features} in layer {target_layer}")

        original_outputs = self._get_model_outputs(sequences)

        if target_layer in self.sae_models:
            hook = self.amplify_features(target_layer, target_features, amplification_strength)
            handle = self.sae_models[target_layer].register_forward_hook(hook)

            try:
                modified_outputs = self._get_model_outputs(sequences)
            finally:
                handle.remove()
        else:
            raise ValueError(f"No SAE model available for layer {target_layer}")

        effect_size = self._compute_output_difference(original_outputs, modified_outputs)
        downstream_effects = self._measure_downstream_effects(
            original_outputs, modified_outputs, target_layer
        )

        return InterventionResult(
            intervention_type="amplification",
            target_layer=target_layer,
            target_features=target_features,
            intervention_strength=amplification_strength,
            original_output=original_outputs,
            modified_output=modified_outputs,
            effect_size=effect_size,
            downstream_effects=downstream_effects
        )

    def test_feature_substitution(self, 
                                sequences: List[str],
                                donor_sequences: List[str],
                                target_layer: int,
                                target_features: List[int]) -> InterventionResult:
        """
        Test substituting feature values from donor sequences.

        Args:
            sequences: Original sequences
            donor_sequences: Sequences to extract feature values from
            target_layer: Layer to perform substitution
            target_features: Features to substitute

        Returns:
            Intervention results
        """
        logger.info(f"Testing feature substitution in layer {target_layer}")

        # Get donor feature values
        donor_outputs = self._get_model_outputs(donor_sequences)
        donor_features = self._extract_sae_features(donor_outputs, target_layer)

        # Create injection values (use mean of donor features)
        feature_values = {}
        for feature_idx in target_features:
            feature_values[feature_idx] = donor_features[:, feature_idx].mean().item()

        original_outputs = self._get_model_outputs(sequences)

        hook = self.inject_features(target_layer, feature_values)
        handle = self.sae_models[target_layer].register_forward_hook(hook)

        try:
            modified_outputs = self._get_model_outputs(sequences)
        finally:
            handle.remove()

        effect_size = self._compute_output_difference(original_outputs, modified_outputs)
        downstream_effects = self._measure_downstream_effects(
            original_outputs, modified_outputs, target_layer
        )

        return InterventionResult(
            intervention_type="substitution",
            target_layer=target_layer,
            target_features=target_features,
            intervention_strength=1.0,
            original_output=original_outputs,
            modified_output=modified_outputs,
            effect_size=effect_size,
            downstream_effects=downstream_effects
        )

    def validate_causal_claim(self, 
                            claim: CausalClaim,
                            test_sequences: List[str],
                            n_trials: int = 10) -> Dict[str, Any]:
        """
        Validate a causal claim through systematic interventions.

        Args:
            claim: Causal claim to test
            test_sequences: Sequences to test on
            n_trials: Number of intervention trials

        Returns:
            Validation results
        """
        logger.info(f"Validating causal claim: {claim.claim_description}")

        results = {
            "claim": claim,
            "validation_success": False,
            "effect_sizes": [],
            "consistency_score": 0.0,
            "trials": []
        }

        for trial in range(n_trials):
            # Test suppression of source features
            suppression_result = self.test_feature_suppression(
                test_sequences, 
                claim.source_layer, 
                claim.source_features,
                suppression_strength=0.0
            )

            # Check effect on target features
            target_effect = self._measure_feature_change(
                suppression_result.original_output,
                suppression_result.modified_output,
                claim.target_layer,
                claim.target_features
            )

            results["trials"].append({
                "trial": trial,
                "intervention_result": suppression_result,
                "target_effect": target_effect
            })

            # Check if effect direction matches expectation
            if claim.expected_effect_direction == "negative":
                expected_sign = target_effect < 0
            elif claim.expected_effect_direction == "positive":
                expected_sign = target_effect > 0
            else:  # bidirectional
                expected_sign = abs(target_effect) > 0.01

            if expected_sign:
                results["effect_sizes"].append(abs(target_effect))

        # Calculate validation success
        if results["effect_sizes"]:
            results["validation_success"] = len(results["effect_sizes"]) / n_trials > 0.7
            results["consistency_score"] = len(results["effect_sizes"]) / n_trials

        return results

    def _get_model_outputs(self, sequences: List[str]) -> Dict[str, Any]:
        """
        Get model outputs for sequences.

        Args:
            sequences: Input sequences

        Returns:
            Model outputs including hidden states and features
        """
        tokenized = self.nucleotide_model.tokenize_sequences(sequences)

        with torch.no_grad():
            outputs = self.nucleotide_model.model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                output_hidden_states=True
            )

        # Extract SAE features for each layer
        layer_features = {}
        for layer_idx, sae in self.sae_models.items():
            if layer_idx < len(outputs.hidden_states):
                hidden_state = outputs.hidden_states[layer_idx]

                # Pool over sequence dimension
                mask = tokenized["attention_mask"].unsqueeze(-1).float()
                pooled_hidden = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

                with torch.no_grad():
                    reconstructed, features = sae(pooled_hidden)

                layer_features[layer_idx] = features

        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "layer_features": layer_features,
            "tokenized": tokenized
        }

    def _extract_sae_features(self, outputs: Dict[str, Any], layer: int) -> torch.Tensor:
        """Extract SAE features for a specific layer."""
        return outputs["layer_features"][layer]

    def _compute_output_difference(self, 
                                 original: Dict[str, Any], 
                                 modified: Dict[str, Any]) -> float:
        """
        Compute the difference between original and modified outputs.

        Args:
            original: Original model outputs
            modified: Modified model outputs

        Returns:
            Effect size (normalized difference)
        """
        # Compare final layer hidden states
        orig_hidden = original["hidden_states"][-1]
        mod_hidden = modified["hidden_states"][-1]

        # Compute normalized difference
        diff = F.mse_loss(mod_hidden, orig_hidden)
        norm = orig_hidden.norm()

        return (diff / (norm + 1e-8)).item()

    def _measure_downstream_effects(self, 
                                  original: Dict[str, Any], 
                                  modified: Dict[str, Any],
                                  intervention_layer: int) -> Dict[str, float]:
        """
        Measure effects on downstream layers.

        Args:
            original: Original outputs
            modified: Modified outputs
            intervention_layer: Layer where intervention was applied

        Returns:
            Dictionary of downstream effects
        """
        downstream_effects = {}

        for layer_idx in range(intervention_layer + 1, len(original["hidden_states"])):
            orig_features = original["layer_features"].get(layer_idx)
            mod_features = modified["layer_features"].get(layer_idx)

            if orig_features is not None and mod_features is not None:
                effect = F.mse_loss(mod_features, orig_features).item()
                downstream_effects[f"layer_{layer_idx}"] = effect

        return downstream_effects

    def _measure_feature_change(self, 
                              original: Dict[str, Any], 
                              modified: Dict[str, Any],
                              layer: int,
                              feature_indices: List[int]) -> float:
        """
        Measure change in specific features.

        Args:
            original: Original outputs
            modified: Modified outputs
            layer: Layer to check
            feature_indices: Features to measure

        Returns:
            Average change in target features
        """
        orig_features = original["layer_features"][layer]
        mod_features = modified["layer_features"][layer]

        orig_target = orig_features[:, feature_indices]
        mod_target = mod_features[:, feature_indices]

        change = (mod_target - orig_target).mean().item()
        return change

class CausalDiscovery:
    """
    Discovers causal relationships between features through systematic interventions.
    """

    def __init__(self, interventioner: FeatureInterventioner):
        """
        Initialize causal discovery.

        Args:
            interventioner: Feature interventioner
        """
        self.interventioner = interventioner

    def discover_feature_dependencies(self, 
                                   sequences: List[str],
                                   source_layer: int,
                                   target_layer: int,
                                   top_k_features: int = 20) -> Dict[Tuple[int, int], float]:
        """
        Discover dependencies between features in different layers.

        Args:
            sequences: Test sequences
            source_layer: Source layer
            target_layer: Target layer
            top_k_features: Number of top features to test

        Returns:
            Dictionary mapping (source_feature, target_feature) to causal strength
        """
        logger.info(f"Discovering dependencies from layer {source_layer} to {target_layer}")

        dependencies = {}

        # Get baseline feature activations
        baseline_outputs = self.interventioner._get_model_outputs(sequences)
        target_features = baseline_outputs["layer_features"][target_layer]

        # Test each source feature
        for source_feature in range(min(top_k_features, 
                                      baseline_outputs["layer_features"][source_layer].shape[1])):

            # Suppress this source feature
            result = self.interventioner.test_feature_suppression(
                sequences, source_layer, [source_feature], suppression_strength=0.0
            )

            # Measure effect on each target feature
            modified_target = result.modified_output["layer_features"][target_layer]

            for target_feature in range(min(top_k_features, target_features.shape[1])):
                orig_activation = target_features[:, target_feature].mean().item()
                mod_activation = modified_target[:, target_feature].mean().item()

                causal_strength = abs(orig_activation - mod_activation)
                dependencies[(source_feature, target_feature)] = causal_strength

        return dependencies

    def find_critical_pathways(self, 
                             sequences: List[str],
                             start_layer: int,
                             end_layer: int,
                             pathway_length: int = 3) -> List[List[Tuple[int, int]]]:
        """
        Find critical computational pathways through the network.

        Args:
            sequences: Test sequences
            start_layer: Starting layer
            end_layer: Ending layer
            pathway_length: Maximum pathway length

        Returns:
            List of pathways (each pathway is a list of (layer, feature) tuples)
        """
        logger.info(f"Finding critical pathways from layer {start_layer} to {end_layer}")

        # This is a simplified implementation
        # In practice, would use more sophisticated path search algorithms

        pathways = []
        layer_range = list(range(start_layer, end_layer + 1))

        if len(layer_range) > pathway_length:
            layer_range = layer_range[:pathway_length]

        # For demonstration, find the most active features at each layer
        baseline_outputs = self.interventioner._get_model_outputs(sequences)

        pathway = []
        for layer in layer_range:
            if layer in baseline_outputs["layer_features"]:
                features = baseline_outputs["layer_features"][layer]
                # Find most active feature
                most_active = features.mean(dim=0).argmax().item()
                pathway.append((layer, most_active))

        if pathway:
            pathways.append(pathway)

        return pathways

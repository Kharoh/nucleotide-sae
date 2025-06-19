"""
Feature Analysis module for interpreting learned SAE features
Based on InterPLM's approach to analyzing protein language model features
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import re
from dataclasses import dataclass
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureInterpretation:
    """
    Interpretation of a learned SAE feature.
    """

    feature_idx: int
    layer: int
    activation_threshold: float
    top_activating_sequences: List[str]
    sequence_motifs: List[str]
    genomic_context: Optional[str] = None
    biological_function: Optional[str] = None
    confidence_score: float = 0.0


@dataclass
class MotifMatch:
    """
    Represents a motif match in a sequence.
    """

    motif: str
    sequence: str
    start_pos: int
    end_pos: int
    score: float


class SequenceMotifAnalyzer:
    """
    Analyzes DNA sequences to identify motifs and patterns.
    """

    def __init__(self):
        # Common DNA motifs and their patterns
        self.known_motifs = {
            "TATA_box": r"TATAAA",
            "CAAT_box": r"CCAAT",
            "GC_box": r"GGGCGG",
            "CpG_island": r"CG.{0,20}CG",
            "Kozak_sequence": r"[AG]CC[AG]CCAUGG",
            "Poly_A_signal": r"AAUAAA",
            "Splice_donor": r"GT[AG]AGT",
            "Splice_acceptor": r"[TC]{10,}[TC]AG",
            "Ribosome_binding": r"AGGAGG",
            "E_box": r"CANNTG",
            "P53_binding": r"PuPuPuC[AT][TA]GPyPyPy",
            "NF_kB_binding": r"GGGRNNYYCC",
            "AP1_binding": r"TGAG?TCA",
            "Homeodomain": r"TAATNN",
            "Heat_shock": r"nGAAnnTTCn",
        }

        # Convert patterns to regex (simplified)
        self.motif_patterns = {}
        for name, pattern in self.known_motifs.items():
            # Simple conversion - in practice would be more sophisticated
            regex_pattern = (
                pattern.replace("N", "[ATCG]")
                .replace("Pu", "[AG]")
                .replace("Py", "[CT]")
            )
            regex_pattern = regex_pattern.replace("R", "[AG]").replace("Y", "[CT]")
            regex_pattern = regex_pattern.replace("S", "[GC]").replace("W", "[AT]")
            regex_pattern = regex_pattern.replace("K", "[GT]").replace("M", "[AC]")
            self.motif_patterns[name] = regex_pattern

    def find_motifs_in_sequence(self, sequence: str) -> List[MotifMatch]:
        """
        Find known motifs in a DNA sequence.

        Args:
            sequence: DNA sequence

        Returns:
            List of motif matches
        """
        matches = []
        sequence = sequence.upper()

        for motif_name, pattern in self.motif_patterns.items():
            try:
                for match in re.finditer(pattern, sequence):
                    matches.append(
                        MotifMatch(
                            motif=motif_name,
                            sequence=sequence,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            score=1.0,  # Simple binary score
                        )
                    )
            except re.error:
                # Skip invalid regex patterns
                continue

        return matches

    def extract_common_subsequences(
        self, sequences: List[str], min_length: int = 4
    ) -> Dict[str, int]:
        """
        Extract common subsequences from a list of sequences.

        Args:
            sequences: List of DNA sequences
            min_length: Minimum subsequence length

        Returns:
            Dictionary of subsequence -> count
        """
        subsequence_counts = Counter()

        for sequence in sequences:
            sequence = sequence.upper()
            for length in range(min_length, min(len(sequence), 10) + 1):
                for i in range(len(sequence) - length + 1):
                    subseq = sequence[i : i + length]
                    subsequence_counts[subseq] += 1

        # Filter by minimum occurrence
        min_count = max(2, len(sequences) // 5)  # At least 20% of sequences
        return {
            subseq: count
            for subseq, count in subsequence_counts.items()
            if count >= min_count
        }

    def analyze_sequence_composition(self, sequences: List[str]) -> Dict[str, float]:
        """
        Analyze the nucleotide composition of sequences.

        Args:
            sequences: List of DNA sequences

        Returns:
            Dictionary of composition statistics
        """
        all_sequences = "".join(sequences).upper()
        total_length = len(all_sequences)

        if total_length == 0:
            return {}

        composition = {
            "A_content": all_sequences.count("A") / total_length,
            "T_content": all_sequences.count("T") / total_length,
            "G_content": all_sequences.count("G") / total_length,
            "C_content": all_sequences.count("C") / total_length,
            "GC_content": (all_sequences.count("G") + all_sequences.count("C"))
            / total_length,
            "AT_content": (all_sequences.count("A") + all_sequences.count("T"))
            / total_length,
        }

        # CpG dinucleotide frequency
        cpg_count = len(re.findall(r"CG", all_sequences))
        composition["CpG_frequency"] = (
            cpg_count / (total_length - 1) if total_length > 1 else 0
        )

        return composition


class FeatureInterpreter:
    """
    Interprets learned SAE features by analyzing their activation patterns.
    """

    def __init__(
        self,
        nucleotide_model,
        sae_models: Dict[int, torch.nn.Module],
        device: str = "cuda",
    ):
        """
        Initialize feature interpreter.

        Args:
            nucleotide_model: Nucleotide transformer model
            sae_models: Dictionary of trained SAE models
            device: Device to run on
        """
        self.nucleotide_model = nucleotide_model
        self.sae_models = sae_models
        self.device = device
        self.motif_analyzer = SequenceMotifAnalyzer()

    def interpret_feature(
        self,
        layer: int,
        feature_idx: int,
        sequences: List[str],
        n_top_sequences: int = 20,
        activation_threshold: float = 0.1,
    ) -> FeatureInterpretation:
        """
        Interpret a specific feature by analyzing sequences that activate it.

        Args:
            layer: Layer index
            feature_idx: Feature index
            sequences: Pool of sequences to test
            n_top_sequences: Number of top activating sequences to analyze
            activation_threshold: Minimum activation to consider

        Returns:
            Feature interpretation
        """
        logger.info(f"Interpreting Layer {layer}, Feature {feature_idx}")

        if layer not in self.sae_models:
            raise ValueError(f"No SAE model available for layer {layer}")

        sae = self.sae_models[layer]
        activations = []
        sequence_activations = []

        # Get activations for all sequences
        for seq in sequences:
            # Get embeddings from the specific layer
            embeddings = self.nucleotide_model.get_embeddings([seq], layer=layer)

            # Get SAE features
            with torch.no_grad():
                _, features = sae(embeddings)
                activation = features[0, feature_idx].item()

            activations.append(activation)
            sequence_activations.append((seq, activation))

        # Sort by activation strength
        sequence_activations.sort(key=lambda x: x[1], reverse=True)

        # Get top activating sequences
        top_sequences = [
            seq
            for seq, act in sequence_activations[:n_top_sequences]
            if act > activation_threshold
        ]

        if not top_sequences:
            logger.warning(
                f"No sequences activated feature {feature_idx} above threshold {activation_threshold}"
            )
            return FeatureInterpretation(
                feature_idx=feature_idx,
                layer=layer,
                activation_threshold=activation_threshold,
                top_activating_sequences=[],
                sequence_motifs=[],
                confidence_score=0.0,
            )

        # Analyze motifs in top sequences
        all_motifs = []
        for seq in top_sequences:
            motifs = self.motif_analyzer.find_motifs_in_sequence(seq)
            all_motifs.extend([m.motif for m in motifs])

        # Find common motifs
        motif_counts = Counter(all_motifs)
        common_motifs = [motif for motif, count in motif_counts.most_common(5)]

        # Extract common subsequences
        common_subseqs = self.motif_analyzer.extract_common_subsequences(top_sequences)
        top_subseqs = list(
            dict(
                sorted(common_subseqs.items(), key=lambda x: x[1], reverse=True)[:5]
            ).keys()
        )

        # Analyze sequence composition
        composition = self.motif_analyzer.analyze_sequence_composition(top_sequences)

        # Determine biological function (simplified heuristics)
        biological_function = self._infer_biological_function(
            common_motifs, composition
        )

        # Calculate confidence score
        confidence = self._calculate_confidence_score(
            len(top_sequences), motif_counts, composition
        )

        return FeatureInterpretation(
            feature_idx=feature_idx,
            layer=layer,
            activation_threshold=activation_threshold,
            top_activating_sequences=top_sequences,
            sequence_motifs=common_motifs + top_subseqs,
            biological_function=biological_function,
            confidence_score=confidence,
        )

    def _infer_biological_function(
        self, motifs: List[str], composition: Dict[str, float]
    ) -> str:
        """
        Infer biological function based on motifs and composition.
        """
        function_indicators = {
            "transcription_regulation": [
                "TATA_box",
                "CAAT_box",
                "GC_box",
                "E_box",
                "AP1_binding",
            ],
            "translation_control": ["Kozak_sequence", "Ribosome_binding"],
            "RNA_processing": ["Splice_donor", "Splice_acceptor", "Poly_A_signal"],
            "DNA_methylation": ["CpG_island"],
            "stress_response": ["Heat_shock", "P53_binding"],
            "immune_response": ["NF_kB_binding"],
        }

        # Check for specific motif patterns
        for function, indicator_motifs in function_indicators.items():
            if any(motif in motifs for motif in indicator_motifs):
                return function

        # Check composition-based indicators
        if composition.get("GC_content", 0) > 0.7:
            return "CpG_island_or_promoter"
        elif composition.get("AT_content", 0) > 0.7:
            return "AT_rich_regulatory_region"

        return "unknown_function"

    def _calculate_confidence_score(
        self, n_sequences: int, motif_counts: Counter, composition: Dict[str, float]
    ) -> float:
        """
        Calculate confidence score for the interpretation.
        """
        # Base score from number of activating sequences
        base_score = min(n_sequences / 20, 1.0)

        # Boost for consistent motifs
        if motif_counts:
            max_motif_count = motif_counts.most_common(1)[0][1]
            motif_consistency = max_motif_count / n_sequences if n_sequences > 0 else 0
            base_score *= 1 + motif_consistency

        # Boost for distinctive composition
        gc_content = composition.get("GC_content", 0.5)
        if abs(gc_content - 0.5) > 0.2:  # Significantly different from random
            base_score *= 1.2

        return min(base_score, 1.0)

    def interpret_all_features(
        self, layer: int, sequences: List[str], min_activation: float = 0.05
    ) -> List[FeatureInterpretation]:
        """
        Interpret all features in a layer.

        Args:
            layer: Layer to analyze
            sequences: Sequences to test
            min_activation: Minimum activation threshold

        Returns:
            List of feature interpretations
        """
        if layer not in self.sae_models:
            raise ValueError(f"No SAE model for layer {layer}")

        sae = self.sae_models[layer]
        n_features = sae.dictionary_size

        interpretations = []

        for feature_idx in tqdm(range(n_features), desc=f"Interpreting Layer {layer}"):
            try:
                interpretation = self.interpret_feature(
                    layer, feature_idx, sequences, activation_threshold=min_activation
                )

                if (
                    interpretation.confidence_score > 0.1
                ):  # Only keep confident interpretations
                    interpretations.append(interpretation)

            except Exception as e:
                logger.warning(f"Failed to interpret feature {feature_idx}: {e}")
                continue

        logger.info(
            f"Successfully interpreted {len(interpretations)}/{n_features} features"
        )
        return interpretations

    def cluster_features(
        self, interpretations: List[FeatureInterpretation], n_clusters: int = 10
    ) -> Dict[int, List[FeatureInterpretation]]:
        """
        Cluster features based on their interpretations.

        Args:
            interpretations: List of feature interpretations
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping cluster index to interpretations
        """
        if not interpretations:
            return {}

        # Create feature vectors based on motif presence
        all_motifs = set()
        for interp in interpretations:
            all_motifs.update(interp.sequence_motifs)

        motif_list = sorted(list(all_motifs))

        # Create binary feature vectors
        feature_vectors = []
        for interp in interpretations:
            vector = [
                1 if motif in interp.sequence_motifs else 0 for motif in motif_list
            ]
            feature_vectors.append(vector)

        if not feature_vectors or not motif_list:
            return {0: interpretations}

        # Cluster
        feature_matrix = np.array(feature_vectors)
        kmeans = KMeans(
            n_clusters=min(n_clusters, len(interpretations)), random_state=42
        )
        cluster_labels = kmeans.fit_predict(feature_matrix)

        # Group by cluster
        clusters = defaultdict(list)
        for interp, label in zip(interpretations, cluster_labels):
            clusters[label].append(interp)

        return dict(clusters)

    def create_feature_summary(
        self, interpretations: List[FeatureInterpretation]
    ) -> pd.DataFrame:
        """
        Create a summary dataframe of feature interpretations.

        Args:
            interpretations: List of feature interpretations

        Returns:
            Summary dataframe
        """
        summary_data = []

        for interp in interpretations:
            summary_data.append(
                {
                    "layer": interp.layer,
                    "feature_idx": interp.feature_idx,
                    "n_activating_sequences": len(interp.top_activating_sequences),
                    "primary_motifs": ", ".join(interp.sequence_motifs[:3]),
                    "biological_function": interp.biological_function,
                    "confidence_score": interp.confidence_score,
                    "activation_threshold": interp.activation_threshold,
                }
            )

        return pd.DataFrame(summary_data)

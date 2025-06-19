"""
Main experiment runner that orchestrates the entire nucleotide biology analysis pipeline
"""

import torch
import os
import json
import pickle
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from pathlib import Path

# Import our modules
from config import Config
from models.nucleotide_transformer import NucleotideTransformerWrapper
from models.sparse_autoencoder import SparseAutoEncoder, SAETrainer
from analysis.attribution_graphs import CircuitTracer, AttributionGraph
from analysis.feature_analysis import FeatureInterpreter
from analysis.interventions import FeatureInterventioner, CausalDiscovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NucleotideBiologyAnalyzer:
    """
    Main class that orchestrates the complete analysis pipeline.
    """

    def __init__(self, config: Config):
        """
        Initialize the analyzer with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = config.model.device

        # Create output directories
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config.experiment.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.nucleotide_model = None
        self.sae_models = {}
        self.circuit_tracer = None
        self.feature_interpreter = None
        self.interventioner = None

        logger.info(
            f"Initialized NucleotideBiologyAnalyzer with output dir: {self.output_dir}"
        )

    def run_complete_analysis(self, sequences: List[str]) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.

        Args:
            sequences: DNA sequences to analyze

        Returns:
            Complete analysis results
        """
        logger.info("Starting complete nucleotide biology analysis pipeline")

        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "sequences_analyzed": len(sequences),
            "stages": {},
        }

        try:
            # Stage 1: Load model and extract embeddings
            logger.info("Stage 1: Loading model and extracting embeddings")
            self._load_nucleotide_model()
            embeddings_results = self._extract_embeddings(sequences)
            results["stages"]["embeddings"] = embeddings_results

            # Stage 2: Train sparse autoencoders
            logger.info("Stage 2: Training sparse autoencoders")
            sae_results = self._train_sparse_autoencoders(sequences)
            results["stages"]["sparse_autoencoders"] = sae_results

            # Stage 3: Interpret features
            logger.info("Stage 3: Interpreting learned features")
            interpretation_results = self._interpret_features(sequences)
            results["stages"]["feature_interpretation"] = interpretation_results

            # Stage 4: Build attribution graphs
            logger.info("Stage 4: Building attribution graphs")
            attribution_results = self._build_attribution_graphs(sequences)
            results["stages"]["attribution_graphs"] = attribution_results

            # Stage 5: Perform interventions
            logger.info("Stage 5: Performing intervention experiments")
            intervention_results = self._perform_interventions(sequences)
            results["stages"]["interventions"] = intervention_results

            # Stage 6: Generate final report
            logger.info("Stage 6: Generating analysis report")
            report = self._generate_final_report(results)
            results["final_report"] = report

            # Save results
            self._save_results(results)

            logger.info("Complete analysis pipeline finished successfully!")

        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            results["error"] = str(e)
            self._save_results(results)
            raise

        return results

    def _load_nucleotide_model(self):
        """Load the nucleotide transformer model."""
        logger.info(f"Loading {self.config.model.model_name}")

        self.nucleotide_model = NucleotideTransformerWrapper(
            model_name=self.config.model.model_name,
            device=self.device,
            cache_dir=self.config.model.cache_dir,
        )

        logger.info(
            f"Model loaded: {self.nucleotide_model.n_layers} layers, "
            f"{self.nucleotide_model.hidden_size} hidden size"
        )

    def _extract_embeddings(self, sequences: List[str]) -> Dict[str, Any]:
        """Extract embeddings from all layers."""
        logger.info(f"Extracting embeddings from {len(sequences)} sequences")

        # Analyze sequence representations
        analysis = self.nucleotide_model.analyze_sequence_representations(sequences)

        # Save embeddings for SAE training
        embeddings_path = self.output_dir / "embeddings.pkl"
        with open(embeddings_path, "wb") as f:
            pickle.dump(analysis["layer_embeddings"], f)

        return {
            "n_layers": analysis["model_info"]["n_layers"],
            "hidden_size": analysis["model_info"]["hidden_size"],
            "layer_statistics": analysis["layer_statistics"],
            "embeddings_saved": str(embeddings_path),
        }

    def _train_sparse_autoencoders(self, sequences: List[str]) -> Dict[str, Any]:
        """Train sparse autoencoders on each layer."""
        logger.info("Training sparse autoencoders")

        # Load embeddings
        embeddings_path = self.output_dir / "embeddings.pkl"
        with open(embeddings_path, "rb") as f:
            layer_embeddings = pickle.load(f)

        sae_results = {}

        # Train SAE for each layer (or subset of layers)
        layers_to_train = [
            0,
            self.nucleotide_model.n_layers // 2,
            self.nucleotide_model.n_layers - 1,
        ]

        for layer_idx in layers_to_train:
            if layer_idx in layer_embeddings:
                logger.info(f"Training SAE for layer {layer_idx}")

                # Get embeddings for this layer
                embeddings = layer_embeddings[layer_idx]

                # Create SAE
                sae = SparseAutoEncoder(
                    activation_dim=self.nucleotide_model.hidden_size,
                    dictionary_size=self.config.sae.dictionary_size,
                    sparsity_penalty=self.config.sae.sparsity_penalty,
                ).to(self.device)

                # Create trainer
                trainer = SAETrainer(sae, learning_rate=self.config.sae.learning_rate)

                # Create dataloader
                dataset = torch.utils.data.TensorDataset(embeddings)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.config.sae.batch_size, shuffle=True
                )

                # Train
                metrics = trainer.train(dataloader, self.config.sae.n_epochs)

                # Save model
                sae_path = self.output_dir / f"sae_layer_{layer_idx}.pt"
                torch.save(sae.state_dict(), sae_path)

                # Store in memory for later use
                self.sae_models[layer_idx] = sae

                sae_results[layer_idx] = {
                    "final_loss": metrics.total_loss[-1] if metrics.total_loss else 0,
                    "final_sparsity": (
                        metrics.feature_density[-1] if metrics.feature_density else 0
                    ),
                    "explained_variance": (
                        metrics.explained_variance[-1]
                        if metrics.explained_variance
                        else 0
                    ),
                    "model_saved": str(sae_path),
                }

        return sae_results

    def _interpret_features(self, sequences: List[str]) -> Dict[str, Any]:
        """Interpret the learned SAE features."""
        logger.info("Interpreting learned features")

        if not self.sae_models:
            raise ValueError("No SAE models available for interpretation")

        # Initialize feature interpreter
        self.feature_interpreter = FeatureInterpreter(
            self.nucleotide_model, self.sae_models, self.device
        )

        interpretation_results = {}

        for layer_idx in self.sae_models.keys():
            logger.info(f"Interpreting features for layer {layer_idx}")

            # Interpret features
            interpretations = self.feature_interpreter.interpret_all_features(
                layer_idx, sequences, min_activation=0.05
            )

            # Create summary
            summary_df = self.feature_interpreter.create_feature_summary(
                interpretations
            )

            # Save summary
            summary_path = (
                self.output_dir / f"feature_interpretations_layer_{layer_idx}.csv"
            )
            summary_df.to_csv(summary_path, index=False)

            # Cluster features
            clusters = self.feature_interpreter.cluster_features(interpretations)

            interpretation_results[layer_idx] = {
                "n_interpretable_features": len(interpretations),
                "avg_confidence": (
                    summary_df["confidence_score"].mean() if not summary_df.empty else 0
                ),
                "n_clusters": len(clusters),
                "summary_saved": str(summary_path),
            }

        return interpretation_results

    def _build_attribution_graphs(self, sequences: List[str]) -> Dict[str, Any]:
        """Build attribution graphs for circuit tracing."""
        logger.info("Building attribution graphs")

        if not self.sae_models:
            raise ValueError("No SAE models available for attribution analysis")

        # Initialize circuit tracer
        self.circuit_tracer = CircuitTracer(
            self.nucleotide_model, self.sae_models, self.device
        )

        attribution_results = {}

        # Analyze a subset of sequences for computational efficiency
        analysis_sequences = sequences[: min(10, len(sequences))]

        for i, seq in enumerate(analysis_sequences):
            logger.info(f"Tracing sequence {i+1}/{len(analysis_sequences)}")

            # For demonstration, trace to the last layer's most active feature
            target_layer = max(self.sae_models.keys())

            # Get the most active feature in the target layer for this sequence
            embeddings = self.nucleotide_model.get_embeddings([seq], layer=target_layer)
            with torch.no_grad():
                _, features = self.sae_models[target_layer](embeddings)
                target_feature = features[0].argmax().item()

            # Build attribution graph
            attribution_graph = self.circuit_tracer.trace_sequence_processing(
                [seq], target_layer, target_feature
            )

            # Save graph
            graph_path = self.output_dir / f"attribution_graph_seq_{i}.pkl"
            with open(graph_path, "wb") as f:
                pickle.dump(attribution_graph, f)

            attribution_results[f"sequence_{i}"] = {
                "sequence": seq[:50] + "..." if len(seq) > 50 else seq,
                "target_layer": target_layer,
                "target_feature": target_feature,
                "n_nodes": len(attribution_graph.nodes),
                "n_edges": len(attribution_graph.edges),
                "graph_saved": str(graph_path),
            }

        return attribution_results

    def _perform_interventions(self, sequences: List[str]) -> Dict[str, Any]:
        """Perform intervention experiments."""
        logger.info("Performing intervention experiments")

        if not self.sae_models:
            raise ValueError("No SAE models available for interventions")

        # Initialize interventioner
        self.interventioner = FeatureInterventioner(
            self.nucleotide_model, self.sae_models, self.device
        )

        intervention_results = {}

        # Test sequences (subset for efficiency)
        test_sequences = sequences[: min(5, len(sequences))]

        # Perform different types of interventions
        for layer_idx in list(self.sae_models.keys())[:2]:  # Test first 2 layers
            logger.info(f"Testing interventions on layer {layer_idx}")

            # Test suppression of top features
            embeddings = self.nucleotide_model.get_embeddings(
                test_sequences, layer=layer_idx
            )
            with torch.no_grad():
                _, features = self.sae_models[layer_idx](embeddings)
                top_features = features.mean(dim=0).topk(3)[1].tolist()

            # Suppression experiment
            suppression_result = self.interventioner.test_feature_suppression(
                test_sequences, layer_idx, top_features, suppression_strength=0.0
            )

            # Amplification experiment
            amplification_result = self.interventioner.test_feature_amplification(
                test_sequences, layer_idx, top_features, amplification_strength=2.0
            )

            intervention_results[layer_idx] = {
                "suppression_effect": suppression_result.effect_size,
                "amplification_effect": amplification_result.effect_size,
                "tested_features": top_features,
                "n_downstream_effects": len(suppression_result.downstream_effects),
            }

        return intervention_results

    def _generate_final_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final analysis report."""
        logger.info("Generating final analysis report")

        report = {
            "summary": {
                "model_analyzed": self.config.model.model_name,
                "sequences_processed": results["sequences_analyzed"],
                "layers_analyzed": len(self.sae_models),
                "total_features_learned": sum(
                    self.config.sae.dictionary_size for _ in self.sae_models
                ),
                "analysis_timestamp": results["timestamp"],
            },
            "key_findings": [],
            "technical_details": {
                "sae_performance": results["stages"].get("sparse_autoencoders", {}),
                "interpretation_quality": results["stages"].get(
                    "feature_interpretation", {}
                ),
                "intervention_effects": results["stages"].get("interventions", {}),
            },
        }

        # Generate key findings
        if "feature_interpretation" in results["stages"]:
            for layer, interp_data in results["stages"][
                "feature_interpretation"
            ].items():
                if interp_data["n_interpretable_features"] > 0:
                    report["key_findings"].append(
                        f"Layer {layer}: Found {interp_data['n_interpretable_features']} "
                        f"interpretable features with average confidence "
                        f"{interp_data['avg_confidence']:.3f}"
                    )

        if "interventions" in results["stages"]:
            for layer, interv_data in results["stages"]["interventions"].items():
                if interv_data["suppression_effect"] > 0.01:
                    report["key_findings"].append(
                        f"Layer {layer}: Feature suppression caused effect size "
                        f"{interv_data['suppression_effect']:.3f} with "
                        f"{interv_data['n_downstream_effects']} downstream effects"
                    )

        return report

    def _save_results(self, results: Dict[str, Any]):
        """Save complete results to file."""
        results_path = self.output_dir / "complete_analysis_results.json"

        # Convert non-serializable objects to strings
        serializable_results = self._make_serializable(results)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {results_path}")

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, torch.nn.Module)):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            return str(obj)
        else:
            return obj


def main():
    """Main function to run the analysis."""
    # Load configuration
    config = Config()

    # Example DNA sequences for testing
    test_sequences = [
        "ATGCGATCGTAGCTGATCGATCGATCGATCGATCGATCGTATAATGCGATCGTAGCTGATCG",
        "TATAATGCGATCGTAGCTGATCGATCGATCGATCGATCGATCGTAGCTGATCGATCGATCG",
        "ATGCGATCGCCCGGGTAGCTGATCGATCGATCGATCGATCGTAGCTGATCGATCGATCGAT",
        "ATGAAAGGATCCGATCGATCGATCGATCGATCGTAGCTGATCGATCGATCGATCGATCGAT",
        "ATGCGATCGTAGCTGATCGATCGATCGTAAAGATCGTAGCTGATCGATCGATCGATCGATC",
        "CCAATGATCGTAGCTGATCGATCGATCGATCGATCGTAGCTGATCGATCGATCGATCGATC",
        "ATGCGATCGTAGCTGATCGATCGGGCGGGATCGTAGCTGATCGATCGATCGATCGATCGAT",
        "ATGCGATCGTAGCTGATCGATCGATCGATCGATCGTAGCTCAGNTGGATCGATCGATCGAT",
        "ATGCGATCGTAGCTGATCGATCGATCGATCGATCGTAGCTGAAUAAATCGATCGATCGATC",
        "ATGCGATCGTAGCTGATCGATCGATCGATCGATCGTAGCTGATCAGGATCCGATCGATCG",
    ]

    # Initialize analyzer
    analyzer = NucleotideBiologyAnalyzer(config)

    # Run complete analysis
    try:
        results = analyzer.run_complete_analysis(test_sequences)
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {analyzer.output_dir}")

        # Print summary
        if "final_report" in results:
            print("\nKey Findings:")
            for finding in results["final_report"]["key_findings"]:
                print(f"  - {finding}")

    except Exception as e:
        print(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()

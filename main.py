#!/usr/bin/env python3
"""
Main script for running Nucleotide Biology Analysis Pipeline
Based on interPLM and Anthropic's "On the Biology of a Large Language Model" techniques

Usage:
    python main.py --config config.json --sequences sequences.txt --output ./results/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from experiments.experiment_runner import NucleotideBiologyAnalyzer
from utils.data_processing import SequenceDataset, SequenceGenerator
from utils.visualization import ComprehensiveVisualizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sequences_from_file(file_path: str) -> List[str]:
    """
    Load DNA sequences from a text file.

    Args:
        file_path: Path to file containing sequences (one per line)

    Returns:
        List of DNA sequences
    """
    sequences = []

    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # Skip comments
                    sequences.append(line)

        logger.info(f"Loaded {len(sequences)} sequences from {file_path}")

    except FileNotFoundError:
        logger.error(f"Sequences file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading sequences: {e}")
        raise

    return sequences


def create_sample_sequences(n_sequences: int = 20, length: int = 200) -> List[str]:
    """
    Create sample DNA sequences for testing.

    Args:
        n_sequences: Number of sequences to generate
        length: Length of each sequence

    Returns:
        List of sample sequences
    """
    logger.info(f"Creating {n_sequences} sample sequences of length {length}")

    sequences = []

    # Create diverse sequence types
    for i in range(n_sequences):
        if i % 4 == 0:
            # Promoter-like sequence with TATA box
            seq = SequenceGenerator.generate_promoter_like_sequence(length)
        elif i % 4 == 1:
            # High GC content sequence
            seq = SequenceGenerator.random_sequence(length, gc_content=0.7)
        elif i % 4 == 2:
            # AT-rich sequence
            seq = SequenceGenerator.random_sequence(length, gc_content=0.3)
        else:
            # Random sequence with motifs
            base_seq = SequenceGenerator.random_sequence(length, gc_content=0.5)
            # Add some known motifs
            motifs = ["TATAAA", "CCAAT", "GGGCGG", "AAUAAA"]
            motif = motifs[i % len(motifs)]
            seq = SequenceGenerator.sequence_with_motif(length, motif)

        sequences.append(seq)

    return sequences


def validate_config(config: Config) -> bool:
    """
    Validate configuration settings.

    Args:
        config: Configuration object

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required fields
        if not config.model.model_name:
            logger.error("Model name not specified in config")
            return False

        if config.sae.dictionary_size <= 0:
            logger.error("Invalid SAE dictionary size")
            return False

        if config.sae.n_epochs <= 0:
            logger.error("Invalid number of training epochs")
            return False

        # Create output directories
        Path(config.experiment.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.experiment.log_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def run_quick_demo() -> Dict[str, Any]:
    """
    Run a quick demonstration of the pipeline with synthetic data.

    Returns:
        Demo results
    """
    logger.info("Running quick demonstration...")

    # Create configuration for demo
    config = Config()
    config.experiment.n_sequences_analysis = 5  # Reduced for demo
    config.sae.n_epochs = 5  # Fewer epochs for demo
    config.sae.dictionary_size = 1024  # Smaller dictionary

    # Create sample sequences
    sequences = create_sample_sequences(n_sequences=5, length=100)

    # Initialize analyzer
    analyzer = NucleotideBiologyAnalyzer(config)

    try:
        # Run analysis
        results = analyzer.run_complete_analysis(sequences)

        logger.info("Demo completed successfully!")
        return results

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


def run_full_analysis(sequences: List[str], config: Config) -> Dict[str, Any]:
    """
    Run the complete analysis pipeline.

    Args:
        sequences: DNA sequences to analyze
        config: Configuration object

    Returns:
        Analysis results
    """
    logger.info("Starting full analysis pipeline...")

    if not validate_config(config):
        raise ValueError("Invalid configuration")

    # Initialize analyzer
    analyzer = NucleotideBiologyAnalyzer(config)

    try:
        # Run complete analysis
        results = analyzer.run_complete_analysis(sequences)

        # Create visualizations
        visualizer = ComprehensiveVisualizer()
        figures = visualizer.create_analysis_dashboard(results)

        # Save visualizations
        output_dir = Path(config.experiment.output_dir)
        figure_paths = []

        for i, fig in enumerate(figures):
            fig_path = output_dir / f"analysis_figure_{i:03d}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            figure_paths.append(str(fig_path))

        results["visualization_paths"] = figure_paths

        logger.info("Full analysis completed successfully!")
        return results

    except Exception as e:
        logger.error(f"Full analysis failed: {e}")
        raise


def create_analysis_report(results: Dict[str, Any], output_path: str):
    """
    Create a markdown report of the analysis results.

    Args:
        results: Analysis results
        output_path: Path to save the report
    """
    logger.info(f"Creating analysis report at {output_path}")

    report_content = f"""# Nucleotide Biology Analysis Report

## Analysis Summary

- **Model Analyzed**: {results.get('config', {}).get('model', {}).get('model_name', 'Unknown')}
- **Sequences Processed**: {results.get('sequences_analyzed', 0)}
- **Analysis Timestamp**: {results.get('timestamp', 'Unknown')}
- **Status**: {"Success" if 'error' not in results else 'Failed'}

## Pipeline Stages

### 1. Embedding Extraction
"""

    if "stages" in results and "embeddings" in results["stages"]:
        embeddings_info = results["stages"]["embeddings"]
        report_content += f"""
- **Model Layers**: {embeddings_info.get('n_layers', 'Unknown')}
- **Hidden Size**: {embeddings_info.get('hidden_size', 'Unknown')}
- **Embeddings Saved**: {embeddings_info.get('embeddings_saved', 'No')}
"""

    report_content += """
### 2. Sparse Autoencoder Training
"""

    if "stages" in results and "sparse_autoencoders" in results["stages"]:
        sae_info = results["stages"]["sparse_autoencoders"]
        for layer, layer_info in sae_info.items():
            report_content += f"""
**Layer {layer}**:
- Final Loss: {layer_info.get('final_loss', 'Unknown'):.4f}
- Final Sparsity: {layer_info.get('final_sparsity', 'Unknown'):.4f}
- Explained Variance: {layer_info.get('explained_variance', 'Unknown'):.4f}
"""

    report_content += """
### 3. Feature Interpretation
"""

    if "stages" in results and "feature_interpretation" in results["stages"]:
        interp_info = results["stages"]["feature_interpretation"]
        for layer, layer_info in interp_info.items():
            report_content += f"""
**Layer {layer}**:
- Interpretable Features: {layer_info.get('n_interpretable_features', 0)}
- Average Confidence: {layer_info.get('avg_confidence', 0):.3f}
- Feature Clusters: {layer_info.get('n_clusters', 0)}
"""

    report_content += """
### 4. Attribution Graph Analysis
"""

    if "stages" in results and "attribution_graphs" in results["stages"]:
        attr_info = results["stages"]["attribution_graphs"]
        for seq_id, seq_info in attr_info.items():
            report_content += f"""
**{seq_id}**:
- Target Layer: {seq_info.get('target_layer', 'Unknown')}
- Target Feature: {seq_info.get('target_feature', 'Unknown')}
- Graph Nodes: {seq_info.get('n_nodes', 0)}
- Graph Edges: {seq_info.get('n_edges', 0)}
"""

    report_content += """
### 5. Intervention Experiments
"""

    if "stages" in results and "interventions" in results["stages"]:
        interv_info = results["stages"]["interventions"]
        for layer, layer_info in interv_info.items():
            report_content += f"""
**Layer {layer}**:
- Suppression Effect: {layer_info.get('suppression_effect', 0):.4f}
- Amplification Effect: {layer_info.get('amplification_effect', 0):.4f}
- Downstream Effects: {layer_info.get('n_downstream_effects', 0)}
"""

    if "final_report" in results and "key_findings" in results["final_report"]:
        report_content += """
## Key Findings

"""
        for finding in results["final_report"]["key_findings"]:
            report_content += f"- {finding}\n"

    if "visualization_paths" in results:
        report_content += """
## Generated Visualizations

"""
        for i, path in enumerate(results["visualization_paths"]):
            report_content += f"- Figure {i+1}: {path}\n"

    report_content += """
## Technical Details

This analysis was performed using a pipeline based on:
1. **InterPLM**: Sparse autoencoders for interpretable feature extraction
2. **Anthropic's "On the Biology of a Large Language Model"**: Attribution graphs for circuit tracing
3. **Nucleotide Transformer**: 500M parameter model trained on diverse human genomes

### Pipeline Components
- Model loading and embedding extraction
- Sparse autoencoder training for feature discovery
- Feature interpretation through sequence analysis
- Attribution graph construction for circuit tracing
- Intervention experiments for causal validation

For more details, see the complete analysis results and generated figures.
"""

    with open(output_path, "w") as f:
        f.write(report_content)

    logger.info(f"Analysis report saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Nucleotide Biology Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with sample sequences (demo mode)
    python main.py --demo

    # Run with custom sequences file
    python main.py --sequences my_sequences.txt --output ./results/

    # Run with custom configuration
    python main.py --config my_config.json --sequences sequences.txt

    # Generate sample sequences and run analysis
    python main.py --generate-samples 50 --length 250 --output ./analysis_results/
        """,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demonstration with synthetic data",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        help="Path to file containing DNA sequences (one per line)",
    )
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument(
        "--output", type=str, default="./output/", help="Output directory for results"
    )
    parser.add_argument(
        "--generate-samples",
        type=int,
        help="Generate N sample sequences instead of loading from file",
    )
    parser.add_argument(
        "--length", type=int, default=200, help="Length of generated sample sequences"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load or create configuration
        if args.config:
            config = Config.load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = Config()
            logger.info("Using default configuration")

        # Set output directory
        config.experiment.output_dir = args.output

        # Get sequences
        if args.demo:
            logger.info("Running demonstration mode")
            results = run_quick_demo()

        elif args.generate_samples:
            logger.info(f"Generating {args.generate_samples} sample sequences")
            sequences = create_sample_sequences(args.generate_samples, args.length)
            results = run_full_analysis(sequences, config)

        elif args.sequences:
            logger.info(f"Loading sequences from {args.sequences}")
            sequences = load_sequences_from_file(args.sequences)
            results = run_full_analysis(sequences, config)

        else:
            logger.error(
                "No sequences specified. Use --demo, --sequences, or --generate-samples"
            )
            parser.print_help()
            return 1

        # Create analysis report
        report_path = Path(config.experiment.output_dir) / "analysis_report.md"
        create_analysis_report(results, str(report_path))

        # Print summary
        print("\n" + "=" * 60)
        print("NUCLEOTIDE BIOLOGY ANALYSIS COMPLETED")
        print("=" * 60)
        print(f"Output directory: {config.experiment.output_dir}")
        print(f"Analysis report: {report_path}")

        if "final_report" in results and "key_findings" in results["final_report"]:
            print("\nKey Findings:")
            for finding in results["final_report"]["key_findings"]:
                print(f"  • {finding}")

        print("\nFor detailed results, see the generated files and report.")
        return 0

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

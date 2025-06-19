"""
Run demo pipeline from feature analysis step using existing results.
"""
import json
import os
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from experiments.experiment_runner import NucleotideBiologyAnalyzer
from utils.visualization import ComprehensiveVisualizer
from main import create_analysis_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name%s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_PATH = "outputs/complete_analysis_results.json"
OUTPUT_DIR = "outputs/"
REPORT_PATH = Path(OUTPUT_DIR) / "analysis_report_from_feature_analysis.md"


def main():
    # Load config (use default demo config)
    config = Config()
    config.experiment.output_dir = OUTPUT_DIR
    config.experiment.n_sequences_analysis = 5
    config.sae.n_epochs = 5
    config.sae.dictionary_size = 1024

    # Load previous results
    if not Path(RESULTS_PATH).exists():
        logger.error(f"Feature analysis results not found at {RESULTS_PATH}")
        return 1
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    logger.info("Loaded feature analysis results. Proceeding with downstream analysis...")

    # Initialize analyzer and load model/SAEs
    analyzer = NucleotideBiologyAnalyzer(config)
    analyzer._load_nucleotide_model()

    # Load SAE models for each layer from disk
    import torch
    from models.sparse_autoencoder import SparseAutoEncoder
    sae_models = {}
    for layer_csv in [f for f in os.listdir(OUTPUT_DIR) if f.startswith("sae_layer_") and f.endswith(".pt")]:
        layer_idx = int(layer_csv.split("_")[2].split(".")[0])
        sae = SparseAutoEncoder(
            activation_dim=analyzer.nucleotide_model.hidden_size,
            dictionary_size=config.sae.dictionary_size,
            sparsity_penalty=config.sae.sparsity_penalty,
        ).to(config.model.device)
        sae.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, layer_csv), map_location=config.model.device))
        sae.eval()
        sae_models[layer_idx] = sae
    analyzer.sae_models = sae_models

    # Set up circuit tracer and interventioner
    from analysis.attribution_graphs import CircuitTracer
    from analysis.interventions import FeatureInterventioner
    analyzer.circuit_tracer = CircuitTracer(analyzer.nucleotide_model, analyzer.sae_models, config.model.device)
    analyzer.interventioner = FeatureInterventioner(analyzer.nucleotide_model, analyzer.sae_models, config.model.device)

    # Get sequences (from results or from embeddings)
    sequences = results.get("sequences", None)
    if not sequences:
        # Try to get from embeddings or fallback
        sequences = results.get("stages", {}).get("embeddings", {}).get("sequences", [])
    if not sequences:
        # Fallback: regenerate demo sequences as in main.py demo mode
        from utils.data_processing import SequenceGenerator
        def create_sample_sequences(n_sequences: int = 20, length: int = 200):
            sequences = []
            for i in range(n_sequences):
                if i % 4 == 0:
                    seq = SequenceGenerator.generate_promoter_like_sequence(length)
                elif i % 4 == 1:
                    seq = SequenceGenerator.random_sequence(length, gc_content=0.7)
                elif i % 4 == 2:
                    seq = SequenceGenerator.random_sequence(length, gc_content=0.3)
                else:
                    base_seq = SequenceGenerator.random_sequence(length, gc_content=0.5)
                    motifs = ["TATAAA", "CCAAT", "GGGCGG", "AAUAAA"]
                    motif = motifs[i % len(motifs)]
                    seq = SequenceGenerator.sequence_with_motif(length, motif)
                sequences.append(seq)
            return sequences
        sequences = create_sample_sequences(n_sequences=5, length=100)
        logger.warning("Sequences not found in results. Regenerated demo sequences for downstream analysis.")

    # Attribution Graphs (Stage 4)
    logger.info("Building attribution graphs...")
    attribution_results = analyzer._build_attribution_graphs(sequences)
    results["stages"]["attribution_graphs"] = attribution_results

    # Interventions (Stage 5)
    logger.info("Performing intervention experiments...")
    intervention_results = analyzer._perform_interventions(sequences)
    results["stages"]["interventions"] = intervention_results

    # Final report (Stage 6)
    logger.info("Generating final analysis report...")
    final_report = analyzer._generate_final_report(results)
    results["final_report"] = final_report

    # Visualization
    visualizer = ComprehensiveVisualizer()
    figures = visualizer.create_analysis_dashboard(results)
    figure_paths = []
    for i, fig in enumerate(figures):
        fig_path = Path(OUTPUT_DIR) / f"analysis_figure_{i:03d}_from_feature_analysis.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        figure_paths.append(str(fig_path))
    results["visualization_paths"] = figure_paths

    # Save updated results
    with open(Path(OUTPUT_DIR) / "complete_analysis_results_from_feature_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create report
    create_analysis_report(results, str(REPORT_PATH))

    print("\n" + "=" * 60)
    print("NUCLEOTIDE BIOLOGY ANALYSIS (FROM FEATURE ANALYSIS) COMPLETED")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Analysis report: {REPORT_PATH}")
    if "final_report" in results and "key_findings" in results["final_report"]:
        print("\nKey Findings:")
        for finding in results["final_report"]["key_findings"]:
            print(f"  • {finding}")
    print("\nFor detailed results, see the generated files and report.")
    return 0

if __name__ == "__main__":
    exit(main())

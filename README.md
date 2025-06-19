# Nucleotide Biology Analysis Pipeline

A comprehensive Python pipeline that applies **interPLM** and **Anthropic's "On the Biology of a Large Language Model"** techniques to analyze the **nucleotide-transformer-500m-1000g** model. This project implements sparse autoencoders for feature discovery and attribution graphs for circuit tracing in genomic sequence analysis.

## Overview

This pipeline combines cutting-edge interpretability techniques from protein and language model research to understand how nucleotide transformers process DNA sequences:

- **Sparse Autoencoders (SAEs)**: Extract interpretable features from transformer hidden states
- **Attribution Graphs**: Trace computational pathways through the network
- **Feature Interpretation**: Analyze learned features for biological meaning
- **Intervention Experiments**: Validate causal hypotheses through targeted manipulations

## Key Features

### 🧬 Model Analysis
- **Nucleotide Transformer Integration**: Full support for the 500M parameter model
- **Multi-layer Analysis**: Extract and analyze features from all transformer layers
- **Sequence Processing**: Handle DNA sequences with proper tokenization and validation

### 🔍 Interpretability Tools
- **Sparse Dictionary Learning**: Train SAEs to decompose neuron activations
- **Circuit Tracing**: Build attribution graphs showing information flow
- **Feature Visualization**: Comprehensive plotting and analysis tools
- **Motif Detection**: Identify biological motifs in feature-activating sequences

### 🧪 Experimental Framework
- **Intervention Studies**: Test causal relationships through feature manipulation
- **Hypothesis Validation**: Systematic testing of computational hypotheses
- **Causal Discovery**: Automated discovery of feature dependencies

### 📊 Analysis & Reporting
- **Interactive Visualizations**: Rich plotting with matplotlib and plotly
- **Comprehensive Reports**: Automated markdown report generation
- **Data Export**: Save results in multiple formats for further analysis

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. **Clone the repository** (or create the project structure):
```bash
mkdir nucleotide_biology_analysis
cd nucleotide_biology_analysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python main.py --demo
```

## Quick Start

### Demo Mode
Run a quick demonstration with synthetic sequences:
```bash
python main.py --demo
```

### Generate Sample Data
Create and analyze synthetic sequences:
```bash
python main.py --generate-samples 50 --length 200 --output ./results/
```

### Analyze Custom Sequences
Provide your own DNA sequences in a text file (one sequence per line):
```bash
python main.py --sequences my_sequences.txt --output ./analysis_output/
```

## Project Structure

```
nucleotide_biology_analysis/
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
├── config.py                 # Configuration management
├── models/
│   ├── nucleotide_transformer.py  # Model wrapper
│   └── sparse_autoencoder.py      # SAE implementation
├── analysis/
│   ├── attribution_graphs.py      # Circuit tracing
│   ├── feature_analysis.py        # Feature interpretation
│   └── interventions.py           # Causal experiments
├── utils/
│   ├── data_processing.py          # Sequence utilities
│   └── visualization.py           # Plotting functions
└── experiments/
    └── experiment_runner.py       # Pipeline orchestration
```

## Usage Examples

### 1. Basic Analysis
```python
from config import Config
from experiment_runner import NucleotideBiologyAnalyzer

# Initialize with default configuration
config = Config()
analyzer = NucleotideBiologyAnalyzer(config)

# Analyze sequences
sequences = ["ATGCGATCGTAGCTGATCGAT...", ...]
results = analyzer.run_complete_analysis(sequences)
```

### 2. Custom Configuration
```python
# Modify configuration
config = Config()
config.sae.dictionary_size = 8192
config.sae.sparsity_penalty = 1e-3
config.experiment.n_sequences_analysis = 500

# Run analysis
analyzer = NucleotideBiologyAnalyzer(config)
results = analyzer.run_complete_analysis(sequences)
```

### 3. Feature Interpretation
```python
from feature_analysis import FeatureInterpreter

interpreter = FeatureInterpreter(nucleotide_model, sae_models)

# Interpret specific features
interpretation = interpreter.interpret_feature(
    layer=12, 
    feature_idx=256, 
    sequences=test_sequences
)

print(f"Biological function: {interpretation.biological_function}")
print(f"Confidence: {interpretation.confidence_score:.3f}")
```

### 4. Intervention Experiments
```python
from interventions import FeatureInterventioner

interventioner = FeatureInterventioner(nucleotide_model, sae_models)

# Test feature suppression
result = interventioner.test_feature_suppression(
    sequences=test_sequences,
    target_layer=10,
    target_features=[100, 150, 200],
    suppression_strength=0.0
)

print(f"Effect size: {result.effect_size:.4f}")
```

## Configuration

The pipeline is highly configurable through the `Config` class:

### Model Configuration
- `model_name`: HuggingFace model identifier
- `max_length`: Maximum sequence length
- `device`: Computing device (cuda/cpu)

### SAE Configuration
- `dictionary_size`: Number of learned features
- `sparsity_penalty`: L1 penalty strength
- `learning_rate`: Training learning rate
- `n_epochs`: Training epochs

### Analysis Configuration
- `n_sequences_analysis`: Number of sequences to analyze
- `intervention_strength`: Strength of feature interventions
- `top_k_features`: Number of top features to analyze

## Methodology

### 1. Sparse Autoencoder Training
The pipeline trains SAEs on transformer hidden states to learn interpretable features:

```
Hidden State → Encoder → Sparse Features → Decoder → Reconstructed State
              (ReLU)                      (Linear)
```

**Loss Function**: `L_total = L_reconstruction + λ * L_sparsity`

### 2. Attribution Graph Construction
Following Anthropic's approach, the pipeline builds attribution graphs showing causal relationships:

- **Nodes**: Interpretable features with activation scores
- **Edges**: Causal influences between features
- **Pruning**: Keep only significant relationships

### 3. Feature Interpretation
Features are interpreted by analyzing sequences that activate them:

- **Motif Detection**: Identify known regulatory motifs
- **Sequence Composition**: Analyze nucleotide composition
- **Biological Function**: Infer likely biological roles

### 4. Intervention Validation
Causal claims are tested through targeted interventions:

- **Suppression**: Set feature activations to zero
- **Amplification**: Increase feature activations
- **Substitution**: Replace with activations from other sequences

## Results and Outputs

The pipeline generates comprehensive outputs:

### Files Generated
- `complete_analysis_results.json`: Full analysis results
- `feature_interpretations_layer_X.csv`: Feature interpretation summaries
- `sae_layer_X.pt`: Trained SAE model weights
- `attribution_graph_seq_X.pkl`: Attribution graphs for sequences
- `analysis_report.md`: Human-readable analysis report

### Visualizations
- SAE training metrics and convergence plots
- Feature activation heatmaps and distributions
- Attribution graph network diagrams
- Intervention effect summaries
- Feature interpretation confidence scores

## Advanced Usage

### Custom Sequence Datasets
```python
from data_processing import SequenceDataset

# Create dataset from sequences
dataset = SequenceDataset(sequences, labels=region_types)

# Filter by length
filtered = dataset.filter_by_length(min_length=100, max_length=500)

# Split train/test
train_dataset, test_dataset = dataset.split_train_test(test_ratio=0.2)
```

### Batch Processing
```python
from utils.data_processing import batch_process_sequences

def analyze_batch(batch_sequences):
    return analyzer.analyze_sequences(batch_sequences)

results = batch_process_sequences(
    large_sequence_list, 
    analyze_batch, 
    batch_size=64
)
```

### Interactive Visualizations
```python
from visualization import ComprehensiveVisualizer

visualizer = ComprehensiveVisualizer()
interactive_fig = visualizer.create_interactive_dashboard(results)
interactive_fig.show()
```

## Research Applications

### Genomic Sequence Analysis
- **Promoter Recognition**: Identify transcriptional start sites
- **Enhancer Detection**: Find regulatory enhancer elements
- **Splice Site Prediction**: Analyze RNA processing signals
- **Motif Discovery**: Uncover novel regulatory motifs

### Model Interpretability
- **Circuit Discovery**: Map computational pathways
- **Feature Analysis**: Understand learned representations
- **Causal Relationships**: Validate mechanistic hypotheses
- **Cross-layer Interactions**: Trace information flow

### Comparative Studies
- **Model Comparison**: Compare different transformer architectures
- **Species Analysis**: Study cross-species sequence patterns
- **Evolution Tracking**: Analyze evolutionary conservation
- **Functional Annotation**: Improve genome annotation

## Performance Considerations

### Memory Requirements
- **Model Loading**: ~2GB for nucleotide-transformer-500m-1000g
- **SAE Training**: ~1GB per layer (depends on dictionary size)
- **Analysis**: ~500MB per 1000 sequences

### Computational Time
- **SAE Training**: ~30 minutes per layer (GPU recommended)
- **Feature Interpretation**: ~10 minutes per layer
- **Attribution Graphs**: ~5 minutes per sequence
- **Full Pipeline**: ~2-4 hours for 100 sequences

### Optimization Tips
- Use GPU acceleration for all training
- Process sequences in batches
- Cache embeddings for multiple analyses
- Use smaller dictionary sizes for initial exploration

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```python
# Reduce batch size or dictionary size
config.sae.batch_size = 256
config.sae.dictionary_size = 4096
```

**Model Download Failures**:
```python
# Set cache directory and retry
config.model.cache_dir = "./model_cache"
```

**Poor Feature Interpretability**:
```python
# Adjust sparsity penalty
config.sae.sparsity_penalty = 1e-3  # Lower for more active features
```

### Debug Mode
```bash
python main.py --demo --verbose
```

## Contributing

We welcome contributions! Areas for development:

- **New Analysis Methods**: Additional interpretability techniques
- **Visualization Improvements**: Enhanced plotting capabilities
- **Performance Optimization**: Faster processing algorithms
- **Biological Integration**: Better motif databases and annotations

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{nucleotide_biology_analysis,
  title={Nucleotide Biology Analysis Pipeline},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/nucleotide-biology-analysis}
}
```

Also cite the foundational works:
- InterPLM: Discovering Interpretable Features in Protein Language Models
- Anthropic: On the Biology of a Large Language Model
- InstaDeep: The Nucleotide Transformer

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Anthropic** for circuit tracing methodology
- **Stanford University** for InterPLM framework
- **InstaDeep, NVIDIA, TUM** for Nucleotide Transformer
- **HuggingFace** for model hosting and transformers library

---

For questions, issues, or contributions, please visit our GitHub repository or contact the maintainers.
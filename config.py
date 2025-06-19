"""
Configuration file for Nucleotide Biology Analysis pipeline
Based on interPLM and Anthropic's "On the Biology of a Large Language Model" techniques
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for the Nucleotide Transformer model"""

    model_name: str = "InstaDeepAI/nucleotide-transformer-500m-1000g"
    max_length: int = 1000  # Maximum sequence length
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: Optional[str] = "./model_cache"


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder"""

    dictionary_size: int = 16384  # Size of the learned dictionary
    sparsity_penalty: float = 1e-4  # L1 penalty coefficient
    learning_rate: float = 3e-4
    batch_size: int = 1024
    n_epochs: int = 50
    validation_split: float = 0.1
    feature_threshold: float = 1e-6  # Threshold for considering a feature "active"


@dataclass
class AttributionConfig:
    """Configuration for Attribution Graph analysis"""

    n_attribution_steps: int = 1000  # Number of steps for gradient-based attribution
    attribution_method: str = "integrated_gradients"  # or "gradient_shap", "saliency"
    baseline_method: str = "zero"  # Baseline for attribution analysis
    noise_level: float = 0.1  # For noisy baselines


@dataclass
class ExperimentConfig:
    """Configuration for experiments and analysis"""

    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    n_sequences_analysis: int = 1000  # Number of sequences for initial analysis
    intervention_strength: float = 2.0  # Strength of feature interventions
    top_k_features: int = 50  # Number of top features to analyze
    random_seed: int = 42


@dataclass
class VisualizationConfig:
    """Configuration for visualization and plotting"""

    figure_size: tuple = (12, 8)
    dpi: int = 300
    color_palette: str = "viridis"
    save_format: str = "png"
    interactive_plots: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing"""

    sequence_examples: List[str] = None
    motif_annotations: Optional[Dict[str, Any]] = None
    genomic_regions: List[str] = None

    def __post_init__(self):
        if self.sequence_examples is None:
            # Example DNA sequences for testing
            self.sequence_examples = [
                "ATGCGATCGTAGCTGATCGATCGATCGATCG",  # Random sequence
                "TATAATGCGATCGTAGCTGATCGATCGATCG",  # With TATA box
                "ATGCGATCGTAGCTGATCGATCGCCCGGGAT",  # With CpG sites
                "ATGAAAGGATCCGATCGATCGATCGATCGAT",  # With restriction site
                "ATGCGATCGTAGCTGATCGATCGATCGTAAA",  # With poly-A signal
            ]

        if self.genomic_regions is None:
            self.genomic_regions = [
                "promoter",
                "enhancer",
                "exon",
                "intron",
                "3_prime_utr",
                "5_prime_utr",
                "intergenic",
            ]


# Main configuration class that combines all configs
@dataclass
class Config:
    """Main configuration class"""

    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        # Set random seeds for reproducibility
        torch.manual_seed(self.experiment.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.experiment.random_seed)

    def save_config(self, path: str):
        """Save configuration to file"""
        import json

        config_dict = {
            "model": self.model.__dict__,
            "sae": self.sae.__dict__,
            "attribution": self.attribution.__dict__,
            "experiment": self.experiment.__dict__,
            "visualization": self.visualization.__dict__,
            "data": self.data.__dict__,
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(cls, path: str):
        """Load configuration from file"""
        import json

        with open(path, "r") as f:
            config_dict = json.load(f)

        return cls(
            model=ModelConfig(**config_dict["model"]),
            sae=SAEConfig(**config_dict["sae"]),
            attribution=AttributionConfig(**config_dict["attribution"]),
            experiment=ExperimentConfig(**config_dict["experiment"]),
            visualization=VisualizationConfig(**config_dict["visualization"]),
            data=DataConfig(**config_dict["data"]),
        )

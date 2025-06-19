"""
Nucleotide Transformer Model Wrapper
Handles loading and inference with the nucleotide-transformer-500m-1000g model
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NucleotideTransformerWrapper:
    """
    Wrapper for the Nucleotide Transformer model that provides easy access to embeddings
    and hidden states across all layers, similar to how InterPLM works with ESM models.
    """

    def __init__(
        self,
        model_name: str = "InstaDeepAI/nucleotide-transformer-500m-1000g",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the Nucleotide Transformer wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        logger.info(f"Loading {model_name} on {device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, cache_dir=cache_dir, output_hidden_states=True
        ).to(device)

        # Print VRAM usage after model is loaded
        self.print_vram_usage("After model load")

        # Get model configuration
        self.config = self.model.config
        self.n_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        logger.info(
            f"Model loaded: {self.n_layers} layers, {self.hidden_size} hidden size"
        )

    def print_vram_usage(self, message: str = ""): 
        """Print the current and max VRAM used by the model on the selected device."""
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 2
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            logger.info(f"[VRAM] {message} Current: {allocated:.2f} MB, Max: {max_allocated:.2f} MB")
        else:
            logger.info(f"[VRAM] {message} CUDA not available or not using GPU.")

    def tokenize_sequences(
        self, sequences: List[str], max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize DNA sequences for model input.

        Args:
            sequences: List of DNA sequences
            max_length: Maximum sequence length (uses model default if None)

        Returns:
            Dictionary with tokenized inputs
        """
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        tokenized = self.tokenizer.batch_encode_plus(
            sequences,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        # Move to device
        return {k: v.to(self.device) for k, v in tokenized.items()}

    def get_embeddings(
        self, sequences: List[str], layer: int = -1, pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Extract embeddings from specified layer.

        Args:
            sequences: List of DNA sequences
            layer: Layer index (-1 for last layer)
            pooling: Pooling method ("mean", "max", "cls", "none")

        Returns:
            Embeddings tensor
        """
        tokenized = self.tokenize_sequences(sequences)

        with torch.no_grad():
            outputs = self.model(**tokenized)

        # Get hidden states from specified layer
        hidden_states = outputs.hidden_states[layer]  # [batch, seq_len, hidden_size]

        if pooling == "none":
            return hidden_states
        elif pooling == "mean":
            # Mean pooling over sequence length (excluding padding)
            attention_mask = tokenized["attention_mask"].unsqueeze(-1)
            masked_embeddings = hidden_states * attention_mask
            return masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        elif pooling == "max":
            return hidden_states.max(dim=1)[0]
        elif pooling == "cls":
            return hidden_states[:, 0, :]  # First token
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    def get_all_layer_embeddings(
        self, sequences: List[str], pooling: str = "mean"
    ) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from all layers.

        Args:
            sequences: List of DNA sequences
            pooling: Pooling method

        Returns:
            Dictionary mapping layer index to embeddings
        """
        tokenized = self.tokenize_sequences(sequences)

        with torch.no_grad():
            outputs = self.model(**tokenized)

        layer_embeddings = {}

        for layer_idx, hidden_states in enumerate(outputs.hidden_states):
            if pooling == "none":
                embeddings = hidden_states
            elif pooling == "mean":
                attention_mask = tokenized["attention_mask"].unsqueeze(-1)
                masked_embeddings = hidden_states * attention_mask
                embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
            elif pooling == "max":
                embeddings = hidden_states.max(dim=1)[0]
            elif pooling == "cls":
                embeddings = hidden_states[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

            layer_embeddings[layer_idx] = embeddings

        return layer_embeddings

    def get_attention_weights(self, sequences: List[str]) -> Dict[int, torch.Tensor]:
        """
        Extract attention weights from all layers.

        Args:
            sequences: List of DNA sequences

        Returns:
            Dictionary mapping layer index to attention weights
        """
        tokenized = self.tokenize_sequences(sequences)

        with torch.no_grad():
            outputs = self.model(**tokenized, output_attentions=True)

        attention_weights = {}
        for layer_idx, attention in enumerate(outputs.attentions):
            attention_weights[layer_idx] = attention

        return attention_weights

    def analyze_sequence_representations(self, sequences: List[str]) -> Dict[str, any]:
        """
        Comprehensive analysis of sequence representations across all layers.

        Args:
            sequences: List of DNA sequences

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing representations for {len(sequences)} sequences")

        # Get embeddings from all layers, one sequence at a time, and print VRAM usage
        layer_embeddings = {}
        attention_weights = {}
        for idx, seq in enumerate(sequences):
            single_layer_embeddings = self.get_all_layer_embeddings([seq], pooling="mean")
            single_attention_weights = self.get_attention_weights([seq])
            for layer_idx, emb in single_layer_embeddings.items():
                if layer_idx not in layer_embeddings:
                    layer_embeddings[layer_idx] = []
                layer_embeddings[layer_idx].append(emb)
            for layer_idx, attn in single_attention_weights.items():
                if layer_idx not in attention_weights:
                    attention_weights[layer_idx] = []
                attention_weights[layer_idx].append(attn)
            self.print_vram_usage(f"After loading sequence {idx+1}/{len(sequences)}")

        # Stack embeddings and attention weights for each layer
        for layer_idx in layer_embeddings:
            layer_embeddings[layer_idx] = torch.cat(layer_embeddings[layer_idx], dim=0)
        for layer_idx in attention_weights:
            attention_weights[layer_idx] = torch.cat(attention_weights[layer_idx], dim=0)

        # Compute layer-wise statistics
        layer_stats = {}
        for layer_idx, embeddings in layer_embeddings.items():
            layer_stats[layer_idx] = {
                "mean_activation": embeddings.mean().item(),
                "std_activation": embeddings.std().item(),
                "sparsity": (embeddings.abs() < 1e-6).float().mean().item(),
                "norm": embeddings.norm(dim=-1).mean().item(),
            }

        return {
            "layer_embeddings": layer_embeddings,
            "attention_weights": attention_weights,
            "layer_statistics": layer_stats,
            "sequences": sequences,
            "n_sequences": len(sequences),
            "model_info": {
                "n_layers": self.n_layers,
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size,
            },
        }

    def embed_single_sequence(self, sequence: str, layer: int = -1) -> torch.Tensor:
        """
        Embed a single sequence (convenience method).

        Args:
            sequence: DNA sequence
            layer: Layer index

        Returns:
            Embedding tensor
        """
        return self.get_embeddings([sequence], layer=layer)[0]

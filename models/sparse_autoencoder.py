"""
Sparse Autoencoder implementation for Nucleotide Transformer interpretability
Based on the approach from InterPLM and similar to Anthropic's sparse dictionary learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SAETrainingMetrics:
    """Metrics tracked during SAE training"""
    reconstruction_loss: List[float]
    sparsity_loss: List[float] 
    total_loss: List[float]
    feature_density: List[float]  # Average number of active features
    explained_variance: List[float]

class SparseAutoEncoder(nn.Module):
    """
    Sparse Autoencoder for learning interpretable features from transformer activations.

    This implementation follows the approach from Anthropic's interpretability work
    and the InterPLM paper, using L1 sparsity penalty and tied weights.
    """

    def __init__(self, 
                 activation_dim: int, 
                 dictionary_size: int,
                 sparsity_penalty: float = 1e-4,
                 tied_weights: bool = True,
                 normalize_decoder: bool = True):
        """
        Initialize the Sparse Autoencoder.

        Args:
            activation_dim: Dimension of input activations (hidden_size of transformer)
            dictionary_size: Size of learned dictionary (typically larger than activation_dim)
            sparsity_penalty: L1 penalty coefficient for sparsity
            tied_weights: Whether to tie encoder and decoder weights (decoder = encoder.T)
            normalize_decoder: Whether to normalize decoder vectors to unit norm
        """
        super().__init__()

        self.activation_dim = activation_dim
        self.dictionary_size = dictionary_size
        self.sparsity_penalty = sparsity_penalty
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder

        # Encoder: activation_dim -> dictionary_size
        self.encoder = nn.Linear(activation_dim, dictionary_size, bias=True)

        if tied_weights:
            # Decoder is transpose of encoder (no separate parameters)
            self.decoder = None
        else:
            # Separate decoder: dictionary_size -> activation_dim
            self.decoder = nn.Linear(dictionary_size, activation_dim, bias=True)

        # Decoder bias (always separate from encoder)
        self.decoder_bias = nn.Parameter(torch.zeros(activation_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        if not self.tied_weights and self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

        nn.init.zeros_(self.decoder_bias)

        # Normalize decoder weights if specified
        if self.normalize_decoder:
            self._normalize_decoder_weights()

    def _normalize_decoder_weights(self):
        """Normalize decoder weight vectors to unit norm"""
        with torch.no_grad():
            if self.tied_weights:
                # Normalize columns of encoder weight (which become rows of decoder)
                weight = self.encoder.weight.data
                norms = weight.norm(dim=0, keepdim=True)
                weight.div_(norms + 1e-8)
            elif self.decoder is not None:
                # Normalize rows of decoder weight
                weight = self.decoder.weight.data
                norms = weight.norm(dim=1, keepdim=True)
                weight.div_(norms + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse feature representation.

        Args:
            x: Input activations [batch_size, activation_dim]

        Returns:
            Sparse feature activations [batch_size, dictionary_size]
        """
        # Linear transformation followed by ReLU for sparsity
        return F.relu(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.

        Args:
            features: Sparse feature activations [batch_size, dictionary_size]

        Returns:
            Reconstructed activations [batch_size, activation_dim]
        """
        if self.tied_weights:
            # Use transpose of encoder weights
            reconstructed = F.linear(features, self.encoder.weight.t())
        else:
            reconstructed = self.decoder(features)

        return reconstructed + self.decoder_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            x: Input activations [batch_size, activation_dim]

        Returns:
            Tuple of (reconstructed_activations, sparse_features)
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features

    def compute_loss(self, x: torch.Tensor, 
                    reconstructed: torch.Tensor, 
                    features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute SAE loss components.

        Args:
            x: Original activations
            reconstructed: Reconstructed activations
            features: Sparse feature activations

        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, x)

        # Sparsity loss (L1 penalty on features)
        sparsity_loss = features.abs().mean()

        # Total loss
        total_loss = reconstruction_loss + self.sparsity_penalty * sparsity_loss

        return {
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss,
            "total_loss": total_loss
        }

    def get_feature_density(self, features: torch.Tensor, threshold: float = 1e-6) -> float:
        """
        Compute average number of active features per sample.

        Args:
            features: Feature activations
            threshold: Threshold for considering a feature "active"

        Returns:
            Average feature density
        """
        active_features = (features > threshold).float()
        return active_features.sum(dim=1).mean().item()

class SAETrainer:
    """
    Trainer class for Sparse Autoencoder.
    """

    def __init__(self, 
                 sae: SparseAutoEncoder,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 0.0,
                 normalize_decoder_freq: int = 100):
        """
        Initialize SAE trainer.

        Args:
            sae: Sparse autoencoder model
            learning_rate: Learning rate for optimization
            weight_decay: L2 regularization coefficient
            normalize_decoder_freq: Frequency to normalize decoder weights
        """
        self.sae = sae
        self.normalize_decoder_freq = normalize_decoder_freq

        # Optimizer
        self.optimizer = optim.Adam(
            sae.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Training metrics
        self.metrics = SAETrainingMetrics(
            reconstruction_loss=[],
            sparsity_loss=[],
            total_loss=[],
            feature_density=[],
            explained_variance=[]
        )

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of activations

        Returns:
            Dictionary of metrics for this step
        """
        self.sae.train()

        # Forward pass
        reconstructed, features = self.sae(batch)

        # Compute losses
        losses = self.sae.compute_loss(batch, reconstructed, features)

        # Backward pass
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        self.optimizer.step()

        # Normalize decoder weights periodically
        if (len(self.metrics.total_loss) + 1) % self.normalize_decoder_freq == 0:
            self.sae._normalize_decoder_weights()

        # Compute additional metrics
        feature_density = self.sae.get_feature_density(features)
        explained_var = self._compute_explained_variance(batch, reconstructed)

        return {
            "reconstruction_loss": losses["reconstruction_loss"].item(),
            "sparsity_loss": losses["sparsity_loss"].item(),
            "total_loss": losses["total_loss"].item(),
            "feature_density": feature_density,
            "explained_variance": explained_var
        }

    def _compute_explained_variance(self, original: torch.Tensor, 
                                  reconstructed: torch.Tensor) -> float:
        """Compute explained variance (R²)"""
        with torch.no_grad():
            ss_res = ((original - reconstructed) ** 2).sum()
            ss_tot = ((original - original.mean()) ** 2).sum()
            return (1 - ss_res / ss_tot).item()

    def train(self, 
              dataloader: torch.utils.data.DataLoader,
              n_epochs: int,
              validation_dataloader: Optional[torch.utils.data.DataLoader] = None,
              log_freq: int = 100) -> SAETrainingMetrics:
        """
        Train the SAE.

        Args:
            dataloader: Training data loader
            n_epochs: Number of training epochs
            validation_dataloader: Optional validation data loader
            log_freq: Frequency to log training progress

        Returns:
            Training metrics
        """
        logger.info(f"Starting SAE training for {n_epochs} epochs")

        for epoch in range(n_epochs):
            epoch_metrics = {
                "reconstruction_loss": [],
                "sparsity_loss": [],
                "total_loss": [],
                "feature_density": [],
                "explained_variance": []
            }

            # Training loop
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Handle cases where dataloader returns tuples

                metrics = self.train_step(batch)

                # Accumulate metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)

                if (batch_idx + 1) % log_freq == 0:
                    avg_loss = np.mean(epoch_metrics["total_loss"][-log_freq:])
                    avg_density = np.mean(epoch_metrics["feature_density"][-log_freq:])
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                              f"Loss={avg_loss:.4f}, Density={avg_density:.2f}")

            # Store epoch averages
            for key, values in epoch_metrics.items():
                getattr(self.metrics, key).append(np.mean(values))

            # Validation
            if validation_dataloader is not None:
                val_metrics = self.validate(validation_dataloader)
                logger.info(f"Epoch {epoch+1} Validation: "
                          f"Loss={val_metrics['total_loss']:.4f}, "
                          f"ExplVar={val_metrics['explained_variance']:.3f}")

        logger.info("Training completed!")
        return self.metrics

    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate the SAE.

        Args:
            dataloader: Validation data loader

        Returns:
            Validation metrics
        """
        self.sae.eval()

        val_metrics = {
            "reconstruction_loss": [],
            "sparsity_loss": [],
            "total_loss": [],
            "feature_density": [],
            "explained_variance": []
        }

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                reconstructed, features = self.sae(batch)
                losses = self.sae.compute_loss(batch, reconstructed, features)

                val_metrics["reconstruction_loss"].append(losses["reconstruction_loss"].item())
                val_metrics["sparsity_loss"].append(losses["sparsity_loss"].item())
                val_metrics["total_loss"].append(losses["total_loss"].item())
                val_metrics["feature_density"].append(self.sae.get_feature_density(features))
                val_metrics["explained_variance"].append(
                    self._compute_explained_variance(batch, reconstructed)
                )

        return {key: np.mean(values) for key, values in val_metrics.items()}

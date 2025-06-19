"""
Data processing utilities for nucleotide sequence analysis
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import re
from collections import Counter, defaultdict
from Bio import SeqIO
from io import StringIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceProcessor:
    """
    Utilities for processing DNA sequences.
    """

    @staticmethod
    def validate_dna_sequence(sequence: str) -> bool:
        """
        Validate that a sequence contains only valid DNA nucleotides.

        Args:
            sequence: DNA sequence string

        Returns:
            True if valid, False otherwise
        """
        valid_chars = set("ATCGN")
        return all(c.upper() in valid_chars for c in sequence)

    @staticmethod
    def clean_sequence(sequence: str) -> str:
        """
        Clean and standardize a DNA sequence.

        Args:
            sequence: Raw DNA sequence

        Returns:
            Cleaned sequence
        """
        # Remove whitespace and convert to uppercase
        cleaned = re.sub(r'\s+', '', sequence.upper())

        # Remove numbers and special characters except ATCGN
        cleaned = re.sub(r'[^ATCGN]', '', cleaned)

        return cleaned

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """
        Get reverse complement of DNA sequence.

        Args:
            sequence: DNA sequence

        Returns:
            Reverse complement
        """
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        complement = ''.join(complement_map[base] for base in sequence.upper())
        return complement[::-1]

    @staticmethod
    def transcribe_to_rna(sequence: str) -> str:
        """
        Transcribe DNA to RNA.

        Args:
            sequence: DNA sequence

        Returns:
            RNA sequence
        """
        return sequence.upper().replace('T', 'U')

    @staticmethod
    def translate_to_protein(sequence: str, frame: int = 0) -> str:
        """
        Translate DNA sequence to protein.

        Args:
            sequence: DNA sequence
            frame: Reading frame (0, 1, or 2)

        Returns:
            Protein sequence
        """
        genetic_code = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }

        sequence = sequence.upper()[frame:]
        protein = []

        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if len(codon) == 3:
                amino_acid = genetic_code.get(codon, 'X')
                protein.append(amino_acid)
                if amino_acid == '*':  # Stop codon
                    break

        return ''.join(protein)

class SequenceGenerator:
    """
    Generate synthetic DNA sequences for testing.
    """

    @staticmethod
    def random_sequence(length: int, gc_content: float = 0.5) -> str:
        """
        Generate random DNA sequence with specified GC content.

        Args:
            length: Sequence length
            gc_content: GC content (0-1)

        Returns:
            Random DNA sequence
        """
        if not 0 <= gc_content <= 1:
            raise ValueError("GC content must be between 0 and 1")

        # Calculate probabilities
        gc_prob = gc_content / 2  # Equal probability for G and C
        at_prob = (1 - gc_content) / 2  # Equal probability for A and T

        nucleotides = ['A', 'T', 'C', 'G']
        probabilities = [at_prob, at_prob, gc_prob, gc_prob]

        sequence = np.random.choice(nucleotides, size=length, p=probabilities)
        return ''.join(sequence)

    @staticmethod
    def sequence_with_motif(base_length: int, motif: str, motif_position: Optional[int] = None) -> str:
        """
        Generate sequence containing a specific motif.

        Args:
            base_length: Base sequence length
            motif: Motif to insert
            motif_position: Position to insert motif (random if None)

        Returns:
            Sequence with motif
        """
        if len(motif) > base_length:
            raise ValueError("Motif longer than sequence")

        if motif_position is None:
            motif_position = np.random.randint(0, base_length - len(motif) + 1)

        # Generate random sequence
        sequence = SequenceGenerator.random_sequence(base_length)

        # Insert motif
        sequence_list = list(sequence)
        for i, nucleotide in enumerate(motif):
            if motif_position + i < len(sequence_list):
                sequence_list[motif_position + i] = nucleotide

        return ''.join(sequence_list)

    @staticmethod
    def generate_promoter_like_sequence(length: int = 200) -> str:
        """
        Generate a promoter-like sequence with typical features.

        Args:
            length: Sequence length

        Returns:
            Promoter-like sequence
        """
        sequence = SequenceGenerator.random_sequence(length, gc_content=0.6)

        # Add TATA box around position -30
        tata_position = max(0, length - 50)
        sequence = (sequence[:tata_position] + 
                   "TATAAA" + 
                   sequence[tata_position + 6:])

        # Add CAAT box upstream
        if length > 100:
            caat_position = max(0, length - 80)
            sequence = (sequence[:caat_position] + 
                       "CCAAT" + 
                       sequence[caat_position + 5:])

        return sequence

class SequenceDataset:
    """
    Dataset class for handling collections of DNA sequences.
    """

    def __init__(self, sequences: List[str], labels: Optional[List[str]] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize sequence dataset.

        Args:
            sequences: List of DNA sequences
            labels: Optional labels for sequences
            metadata: Optional metadata dictionary
        """
        self.sequences = [SequenceProcessor.clean_sequence(seq) for seq in sequences]
        self.labels = labels
        self.metadata = metadata or {}

        # Validate sequences
        invalid_sequences = []
        for i, seq in enumerate(self.sequences):
            if not SequenceProcessor.validate_dna_sequence(seq):
                invalid_sequences.append(i)

        if invalid_sequences:
            logger.warning(f"Found {len(invalid_sequences)} invalid sequences")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"sequence": self.sequences[idx], "index": idx}

        if self.labels:
            item["label"] = self.labels[idx]

        return item

    def get_sequence_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the sequence dataset.

        Returns:
            Dictionary of statistics
        """
        lengths = [len(seq) for seq in self.sequences]

        # Nucleotide composition
        all_nucleotides = ''.join(self.sequences)
        total_length = len(all_nucleotides)

        composition = {}
        if total_length > 0:
            for nucleotide in 'ATCG':
                composition[f'{nucleotide}_content'] = all_nucleotides.count(nucleotide) / total_length

        return {
            "n_sequences": len(self.sequences),
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "mean_length": np.mean(lengths) if lengths else 0,
            "median_length": np.median(lengths) if lengths else 0,
            "composition": composition
        }

    def filter_by_length(self, min_length: int, max_length: int) -> "SequenceDataset":
        """
        Filter sequences by length.

        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length

        Returns:
            Filtered dataset
        """
        filtered_sequences = []
        filtered_labels = []

        for i, seq in enumerate(self.sequences):
            if min_length <= len(seq) <= max_length:
                filtered_sequences.append(seq)
                if self.labels:
                    filtered_labels.append(self.labels[i])

        return SequenceDataset(
            filtered_sequences, 
            filtered_labels if self.labels else None,
            self.metadata
        )

    def split_train_test(self, test_ratio: float = 0.2) -> Tuple["SequenceDataset", "SequenceDataset"]:
        """
        Split dataset into train and test sets.

        Args:
            test_ratio: Fraction of data for test set

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        n_total = len(self.sequences)
        n_test = int(n_total * test_ratio)

        # Random split
        indices = np.random.permutation(n_total)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        train_sequences = [self.sequences[i] for i in train_indices]
        test_sequences = [self.sequences[i] for i in test_indices]

        train_labels = [self.labels[i] for i in train_indices] if self.labels else None
        test_labels = [self.labels[i] for i in test_indices] if self.labels else None

        train_dataset = SequenceDataset(train_sequences, train_labels, self.metadata)
        test_dataset = SequenceDataset(test_sequences, test_labels, self.metadata)

        return train_dataset, test_dataset

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert dataset to pandas DataFrame.

        Returns:
            DataFrame with sequences and labels
        """
        data = {
            "sequence": self.sequences,
            "length": [len(seq) for seq in self.sequences]
        }

        if self.labels:
            data["label"] = self.labels

        return pd.DataFrame(data)

    @classmethod
    def from_fasta(cls, fasta_content: str) -> "SequenceDataset":
        """
        Create dataset from FASTA format content.

        Args:
            fasta_content: FASTA format string

        Returns:
            SequenceDataset
        """
        sequences = []
        labels = []

        for record in SeqIO.parse(StringIO(fasta_content), "fasta"):
            sequences.append(str(record.seq))
            labels.append(record.id)

        return cls(sequences, labels)

    @classmethod
    def generate_test_dataset(cls, n_sequences: int = 100, 
                            sequence_length: int = 200) -> "SequenceDataset":
        """
        Generate a test dataset with synthetic sequences.

        Args:
            n_sequences: Number of sequences to generate
            sequence_length: Length of each sequence

        Returns:
            Test dataset
        """
        sequences = []
        labels = []

        for i in range(n_sequences):
            if i % 3 == 0:
                # Promoter-like sequence
                seq = SequenceGenerator.generate_promoter_like_sequence(sequence_length)
                label = "promoter"
            elif i % 3 == 1:
                # High GC sequence
                seq = SequenceGenerator.random_sequence(sequence_length, gc_content=0.7)
                label = "high_gc"
            else:
                # Random sequence
                seq = SequenceGenerator.random_sequence(sequence_length, gc_content=0.5)
                label = "random"

            sequences.append(seq)
            labels.append(label)

        return cls(sequences, labels, {"source": "synthetic", "generation_date": "2025"})

# Utility functions
def create_sequence_embeddings_dataset(sequences: List[str], 
                                     embeddings: torch.Tensor) -> torch.utils.data.TensorDataset:
    """
    Create a PyTorch dataset from sequences and embeddings.

    Args:
        sequences: List of DNA sequences
        embeddings: Tensor of embeddings

    Returns:
        TensorDataset
    """
    return torch.utils.data.TensorDataset(embeddings)

def batch_process_sequences(sequences: List[str], 
                          process_fn: callable,
                          batch_size: int = 32) -> List[Any]:
    """
    Process sequences in batches.

    Args:
        sequences: List of sequences to process
        process_fn: Function to apply to each batch
        batch_size: Batch size

    Returns:
        List of processed results
    """
    results = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_result = process_fn(batch)
        results.extend(batch_result if isinstance(batch_result, list) else [batch_result])

    return results

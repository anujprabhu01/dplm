#!/usr/bin/env python3
"""
Preprocess TCR repertoire data for DPLM training.

This script converts TCR sequences from pickle format to DPLM-compatible format.
It creates train/validation splits and prepares the data for model training.

Usage:
    python scripts/preprocess_tcr_data.py \
        --input ~/data/tcr_dataset/tcr_repertoire_seqs.pkl \
        --output data-bin/tcr_2m \
        --max_sequences 2000000 \
        --val_split 0.05
"""

import argparse
import pickle
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tcr_sequences(input_path: Path) -> List[str]:
    """Load TCR sequences from pickle file."""
    logger.info(f"Loading TCR sequences from {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different pickle formats
    if isinstance(data, list):
        sequences = data
    elif isinstance(data, dict):
        # Extract sequences from dict (adjust key based on actual format)
        if 'sequences' in data:
            sequences = data['sequences']
        elif 'seq' in data:
            sequences = data['seq']
        else:
            # Assume values are sequences
            sequences = list(data.values())
    else:
        # Handle pandas DataFrame
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                logger.info(f"Detected pandas DataFrame with columns: {list(data.columns)}")
                # Try common column names for TCR sequences
                possible_cols = ['sequence', 'seq', 'amino_acid', 'cdr3', 'junction', 'junction_aa', 'AASeq']
                seq_col = None
                for col in possible_cols:
                    if col in data.columns:
                        seq_col = col
                        break
                
                if seq_col is None:
                    # Use first column that looks like sequences
                    for col in data.columns:
                        if data[col].dtype == 'object':
                            seq_col = col
                            break
                
                if seq_col is None:
                    raise ValueError(f"Could not find sequence column in DataFrame. Columns: {list(data.columns)}")
                
                logger.info(f"Using column '{seq_col}' for sequences")
                sequences = data[seq_col].dropna().tolist()
            else:
                raise ValueError(f"Unexpected data format: {type(data)}")
        except ImportError:
            raise ValueError(f"Data is pandas DataFrame but pandas not installed")
    
    logger.info(f"Loaded {len(sequences)} sequences")
    return sequences


def filter_sequences(sequences: List[str], 
                     min_len: int = 50, 
                     max_len: int = 512) -> List[str]:
    """Filter sequences by length and validity."""
    logger.info(f"Filtering sequences (min_len={min_len}, max_len={max_len})")
    
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    filtered = []
    
    for seq in tqdm(sequences, desc="Filtering"):
        # Check length
        if not (min_len <= len(seq) <= max_len):
            continue
        
        # Check for valid amino acids only
        if not all(aa in valid_amino_acids for aa in seq.upper()):
            continue
        
        filtered.append(seq.upper())
    
    logger.info(f"Kept {len(filtered)} sequences after filtering")
    return filtered


def create_splits(sequences: List[str], 
                  val_split: float = 0.05,
                  seed: int = 42) -> Dict[str, List[str]]:
    """Create train/validation splits."""
    logger.info(f"Creating splits (val_split={val_split})")
    
    np.random.seed(seed)
    indices = np.random.permutation(len(sequences))
    
    n_val = int(len(sequences) * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    splits = {
        'train': [sequences[i] for i in train_indices],
        'validation': [sequences[i] for i in val_indices]
    }
    
    logger.info(f"Train: {len(splits['train'])} sequences")
    logger.info(f"Validation: {len(splits['validation'])} sequences")
    
    return splits


def save_sequences(sequences: List[str], output_path: Path):
    """Save sequences in text format (one per line)."""
    logger.info(f"Saving {len(sequences)} sequences to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for seq in tqdm(sequences, desc="Writing"):
            f.write(seq + '\n')
    
    logger.info(f"Saved to {output_path}")


def save_metadata(metadata: Dict[str, Any], output_path: Path):
    """Save metadata as JSON."""
    logger.info(f"Saving metadata to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {output_path}")


def compute_statistics(sequences: List[str]) -> Dict[str, Any]:
    """Compute sequence statistics."""
    if len(sequences) == 0:
        return {
            'num_sequences': 0,
            'min_length': 0,
            'max_length': 0,
            'mean_length': 0,
            'median_length': 0,
            'std_length': 0,
            'total_amino_acids': 0
        }
    
    lengths = [len(seq) for seq in sequences]
    
    stats = {
        'num_sequences': len(sequences),
        'min_length': int(np.min(lengths)),
        'max_length': int(np.max(lengths)),
        'mean_length': float(np.mean(lengths)),
        'median_length': float(np.median(lengths)),
        'std_length': float(np.std(lengths)),
        'total_amino_acids': int(np.sum(lengths))
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Preprocess TCR data for DPLM')
    parser.add_argument('--input', type=str, required=True,
                        help='Input pickle file with TCR sequences')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--max_sequences', type=int, default=2_000_000,
                        help='Maximum number of sequences to use')
    parser.add_argument('--val_split', type=float, default=0.05,
                        help='Validation split fraction')
    parser.add_argument('--min_len', type=int, default=5,
                        help='Minimum sequence length')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # Load sequences
    sequences = load_tcr_sequences(input_path)
    
    # Filter sequences
    sequences = filter_sequences(sequences, min_len=args.min_len, max_len=args.max_len)
    
    # Limit to max_sequences
    if len(sequences) > args.max_sequences:
        logger.info(f"Sampling {args.max_sequences} sequences from {len(sequences)}")
        np.random.seed(args.seed)
        indices = np.random.choice(len(sequences), args.max_sequences, replace=False)
        sequences = [sequences[i] for i in indices]
    
    # Create splits
    splits = create_splits(sequences, val_split=args.val_split, seed=args.seed)
    
    # Save sequences
    save_sequences(splits['train'], output_dir / 'train.txt')
    save_sequences(splits['validation'], output_dir / 'validation.txt')
    
    # Compute and save metadata
    metadata = {
        'dataset': 'tcr_repertoires_healthy',
        'source': str(input_path),
        'preprocessing': {
            'min_length': args.min_len,
            'max_length': args.max_len,
            'val_split': args.val_split,
            'seed': args.seed
        },
        'train_stats': compute_statistics(splits['train']),
        'validation_stats': compute_statistics(splits['validation']),
        'total_stats': compute_statistics(sequences)
    }
    
    save_metadata(metadata, output_dir / 'metadata.json')
    
    logger.info("="*50)
    logger.info("Preprocessing complete!")
    logger.info(f"Train sequences: {len(splits['train'])}")
    logger.info(f"Validation sequences: {len(splits['validation'])}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*50)


if __name__ == '__main__':
    main()



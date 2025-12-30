#!/usr/bin/env python3
"""
Create subsets of UniRef50 dataset for scaling law experiments.

This script downloads the UniRef50 dataset from HuggingFace and creates
multiple subsets of different sizes for systematic scaling experiments.
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def create_subsets(
    output_dir="data-bin",
    subset_sizes=None,
    seed=42,
    download_full=False
):
    """
    Create multiple subsets of UniRef50 for scaling experiments.
    
    Args:
        output_dir: Base directory to save subsets
        subset_sizes: List of subset sizes (number of sequences)
        seed: Random seed for reproducibility
        download_full: Whether to also save the full dataset
    """
    if subset_sizes is None:
        # Default sizes for scaling experiments
        subset_sizes = [
            1_000,      # Tiny - for smoke testing
            10_000,     # Small - quick experiments
            100_000,    # Medium-small
            500_000,    # Medium
            1_000_000,  # Large
            5_000_000,  # Very large (if compute allows)
            # 10_000_000, # Huge (uncomment if you have compute budget)
        ]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("📊 Creating UniRef50 Subsets for Scaling Laws")
    print("=" * 80)
    
    # Load dataset from HuggingFace
    print("\n📥 Loading UniRef50 dataset from HuggingFace...")
    print("   This may take a while on first run (downloads ~8GB)...")
    
    try:
        # Load in streaming mode first to check
        dataset = load_dataset(
            "airkingbd/uniref50",
            split="train",
            streaming=False,  # Load full dataset for subsetting
            trust_remote_code=True
        )
        print(f"✅ Dataset loaded: {len(dataset)} sequences")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Install huggingface_hub: pip install huggingface_hub")
        print("3. Ensure enough disk space (~8GB)")
        return
    
    # Shuffle dataset with fixed seed for reproducibility
    print(f"\n🔀 Shuffling dataset (seed={seed})...")
    dataset = dataset.shuffle(seed=seed)
    
    # Create subsets
    print(f"\n✂️  Creating {len(subset_sizes)} subsets...")
    for size in subset_sizes:
        subset_name = f"uniref50_{size // 1000}k" if size >= 1000 else f"uniref50_{size}"
        subset_path = output_dir / subset_name
        
        print(f"\n  📁 Creating subset: {subset_name} ({size:,} sequences)")
        
        # Check if already exists
        if subset_path.exists():
            print(f"     ⚠️  Already exists, skipping...")
            continue
        
        # Create subset
        try:
            subset = dataset.select(range(min(size, len(dataset))))
            
            # Save subset
            subset_path.mkdir(parents=True, exist_ok=True)
            subset.save_to_disk(str(subset_path))
            
            print(f"     ✅ Saved to: {subset_path}")
            print(f"     📊 Size: {len(subset):,} sequences")
            
        except Exception as e:
            print(f"     ❌ Error creating subset: {e}")
            continue
    
    # Optionally save full dataset
    if download_full:
        full_path = output_dir / "uniref50_full"
        if not full_path.exists():
            print(f"\n📦 Saving full dataset to: {full_path}")
            dataset.save_to_disk(str(full_path))
            print(f"   ✅ Saved: {len(dataset):,} sequences")
    
    print("\n" + "=" * 80)
    print("✅ Subset creation complete!")
    print("=" * 80)
    
    # Print summary
    print("\n📋 Summary of created subsets:")
    print("-" * 80)
    for size in subset_sizes:
        subset_name = f"uniref50_{size // 1000}k" if size >= 1000 else f"uniref50_{size}"
        subset_path = output_dir / subset_name
        if subset_path.exists():
            print(f"  ✓ {subset_name:20s} → {subset_path}")
    print("-" * 80)
    
    print("\n🚀 Next steps:")
    print("  1. Run smoke test: bash scripts/run_smoke_test.sh")
    print("  2. Verify training works with tiny dataset")
    print("  3. Run scaling experiments with larger subsets")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create UniRef50 subsets for scaling law experiments"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data-bin",
        help="Output directory for subsets"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Custom subset sizes (e.g., --sizes 1000 10000 100000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--download_full",
        action="store_true",
        help="Also save the full dataset locally"
    )
    
    args = parser.parse_args()
    
    create_subsets(
        output_dir=args.output_dir,
        subset_sizes=args.sizes,
        seed=args.seed,
        download_full=args.download_full
    )


if __name__ == "__main__":
    main()


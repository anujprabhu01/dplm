#!/usr/bin/env python
"""
Smoke test for unconditional generation using a local checkpoint.
Usage:
    python scripts/test_generation_smoke.py --checkpoint_path logs/smoke_test_*/checkpoints/last.ckpt
"""
import argparse
import torch
from pathlib import Path
from pprint import pprint

from byprot.models.dplm.dplm import DiffusionProteinLanguageModel


def generate_from_checkpoint(checkpoint_path, num_seqs=5, seq_len=100, max_iter=100):
    """
    Generate protein sequences from a local checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint (e.g., logs/run_name/checkpoints/last.ckpt)
        num_seqs: Number of sequences to generate
        seq_len: Length of sequences to generate
        max_iter: Maximum diffusion iterations
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Find the .hydra/config.yaml file
    # Structure: logs/run_name/checkpoints/last.ckpt -> logs/run_name/.hydra/config.yaml
    run_dir = checkpoint_path.parents[1]  # Go up from checkpoints/ to run_name/
    hydra_config = run_dir / ".hydra" / "config.yaml"
    
    if not hydra_config.exists():
        print(f"Warning: .hydra/config.yaml not found at {hydra_config}")
        print("Trying to load model from checkpoint directly...")
        # Try loading just the checkpoint and use default config
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        print("Checkpoint keys:", list(ckpt.keys())[:10])
        print("\n❌ Cannot load model without config.yaml")
        print("This checkpoint might need to be loaded differently.")
        return
    
    # Load model using from_pretrained with from_huggingface=False
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = DiffusionProteinLanguageModel.from_pretrained(
        str(checkpoint_path),
        from_huggingface=False
    )
    
    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda()
    device = next(model.parameters()).device
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Device: {device}")
    print(f"   Generating {num_seqs} sequences of length {seq_len}")
    print(f"   Max iterations: {max_iter}\n")
    
    # Initialize generation (all masks) - same as generate_dplm.py
    seq = ["<mask>"] * seq_len
    init_seq = ["".join(seq)] * num_seqs
    batch = tokenizer.batch_encode_plus(
        init_seq,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    input_tokens = batch["input_ids"].to(device)
    
    print(f"Input shape: {input_tokens.shape}")
    print(f"Starting generation...\n")
    
    # Generate
    partial_mask = input_tokens.ne(model.mask_id)  # Positions NOT to mask
    with torch.cuda.amp.autocast():
        outputs = model.generate(
            input_tokens=input_tokens,
            tokenizer=tokenizer,
            max_iter=max_iter,
            sampling_strategy="gumbel_argmax",  # Simple strategy for smoke test
            partial_masks=partial_mask,
        )
    
    # Decode
    output_results = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ]
    
    print("=" * 60)
    print("🎉 GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated {len(output_results)} sequences:\n")
    for idx, seq in enumerate(output_results):
        print(f"Sequence {idx + 1} (length {len(seq)}):")
        print(f"  {seq[:80]}..." if len(seq) > 80 else f"  {seq}")
        print()
    
    # Save to file
    output_dir = Path("generation-results/smoke_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"generated_L{seq_len}_iter{max_iter}.fasta"
    
    with open(output_file, "w") as f:
        for idx, seq in enumerate(output_results):
            f.write(f">SEQUENCE_{idx}_L={len(seq)}\n")
            f.write(f"{seq}\n")
    
    print(f"✅ Sequences saved to: {output_file}")
    print(f"\nNote: These sequences are from a minimally-trained model (100 steps).")
    print(f"      They may not be biologically meaningful yet!")


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test unconditional generation from local checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint (e.g., logs/smoke_test_*/checkpoints/last.ckpt)"
    )
    parser.add_argument("--num_seqs", type=int, default=5, help="Number of sequences to generate")
    parser.add_argument("--seq_len", type=int, default=100, help="Length of sequences")
    parser.add_argument("--max_iter", type=int, default=100, help="Max diffusion iterations")
    
    args = parser.parse_args()
    
    generate_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        num_seqs=args.num_seqs,
        seq_len=args.seq_len,
        max_iter=args.max_iter
    )


if __name__ == "__main__":
    main()


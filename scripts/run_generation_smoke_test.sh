#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=================================="
echo "🧪 DPLM Generation Smoke Test"
echo "=================================="

# Check if conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "dplm" ]]; then
    echo "Error: Conda environment 'dplm' is not active."
    echo "Please activate it first: conda activate dplm"
    exit 1
fi
echo "✓ Conda environment: $CONDA_DEFAULT_ENV"

# Find the most recent checkpoint
CHECKPOINT_DIR="logs"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: No logs directory found. Please run training first."
    exit 1
fi

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "last.ckpt" -type f | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $CHECKPOINT_DIR"
    echo "Please run training first: bash scripts/run_smoke_test.sh"
    exit 1
fi

echo "✓ Found checkpoint: $LATEST_CHECKPOINT"

# Check if .hydra config exists
CHECKPOINT_DIR_PATH=$(dirname "$LATEST_CHECKPOINT")
RUN_DIR=$(dirname "$CHECKPOINT_DIR_PATH")
HYDRA_CONFIG="$RUN_DIR/.hydra/config.yaml"

if [ ! -f "$HYDRA_CONFIG" ]; then
    echo "Warning: .hydra/config.yaml not found at $HYDRA_CONFIG"
    echo "The checkpoint might not load correctly."
    echo "Run directory: $RUN_DIR"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "=================================="
echo "🚀 Starting generation..."
echo "=================================="
echo ""
echo "Configuration:"
echo "  - Checkpoint: $LATEST_CHECKPOINT"
echo "  - Sequences: 5"
echo "  - Length: 100 amino acids"
echo "  - Max iterations: 100 (reduced for smoke test)"
echo ""
echo "This will verify:"
echo "  ✓ Model loads from checkpoint"
echo "  ✓ Generation pipeline works"
echo "  ✓ Output sequences are produced"
echo ""

# Run generation
python scripts/test_generation_smoke.py \
    --checkpoint_path "$LATEST_CHECKPOINT" \
    --num_seqs 5 \
    --seq_len 100 \
    --max_iter 100

echo ""
echo "=================================="
echo "✅ Generation smoke test complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Check generated sequences in generation-results/smoke_test/"
echo "  2. If successful, you can generate longer sequences with more iterations"
echo "  3. For better quality, train the model for more steps first"
echo ""


#!/bin/bash
#
# Smoke test script for DPLM pretraining
# This script runs a minimal training to verify everything works correctly
#
# Usage: bash scripts/run_smoke_test.sh
#

set -e  # Exit on error

echo "=================================="
echo "🧪 DPLM Smoke Test"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "❌ Error: train.py not found. Please run from dplm/ root directory."
    exit 1
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda environment detected."
    echo "   Please activate the dplm environment first:"
    echo "   conda activate dplm"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Check if tiny dataset exists
TINY_DATASET="data-bin/uniref50_1k"
if [ ! -d "$TINY_DATASET" ]; then
    echo "📥 Tiny dataset not found. Creating it now..."
    echo ""
    python scripts/create_uniref50_subsets.py --sizes 1000
    echo ""
fi

echo "✓ Tiny dataset found: $TINY_DATASET"
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU only

# Run smoke test
echo "=================================="
echo "🚀 Starting smoke test training..."
echo "=================================="
echo ""
echo "Configuration:"
echo "  - Model: ESM2-35M (ultra-lightweight)"
echo "  - Dataset: 1,000 sequences"
echo "  - Steps: 100"
echo "  - Batch size: ~512 tokens"
echo "  - Expected time: 5-10 minutes"
echo ""
echo "This will verify:"
echo "  ✓ Dataset loading works"
echo "  ✓ Model initialization works"
echo "  ✓ Training loop runs"
echo "  ✓ Checkpoints save correctly"
echo "  ✓ Logging works"
echo ""

# Run training
python train.py \
    experiment=dplm/dplm_150m_smoke_test \
    name=smoke_test_$(date +%Y%m%d_%H%M%S)

echo ""
echo "=================================="
echo "✅ Smoke test complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Check logs in logs/train/runs/"
echo "  2. Check tensorboard: tensorboard --logdir logs/"
echo "  3. If successful, proceed to scaling experiments"
echo ""
echo "To run actual experiments:"
echo "  bash scripts/run_scaling_experiment.sh"
echo ""


#!/bin/bash
# Run a single model size scaling experiment
# Usage: bash scripts/run_single_scaling_experiment.sh <model_size>
# Example: bash scripts/run_single_scaling_experiment.sh 8m

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_size>"
    echo "Available sizes: 8m, 35m, 150m, 650m"
    exit 1
fi

MODEL_SIZE=$1
DATE=$(date +%Y%m%d_%H%M%S)
CONDA_ENV="dplm"

# Map model size to config and GPU settings
case ${MODEL_SIZE} in
    8m)
        CONFIG="dplm_8m_tcr_2m"
        DEVICES="0"
        ;;
    35m)
        CONFIG="dplm_35m_tcr_2m"
        DEVICES="0"
        ;;
    150m)
        CONFIG="dplm_150m_tcr_2m"
        DEVICES="0,1"
        ;;
    650m)
        CONFIG="dplm_650m_tcr_2m"
        DEVICES="0,1,2,3"
        ;;
    *)
        echo "Error: Unknown model size '${MODEL_SIZE}'"
        echo "Available sizes: 8m, 35m, 150m, 650m"
        exit 1
        ;;
esac

echo "========================================"
echo "DPLM Scaling Experiment: ${MODEL_SIZE}"
echo "Config: ${CONFIG}"
echo "GPUs: ${DEVICES}"
echo "Date: ${DATE}"
echo "========================================"

# Activate conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Check data
if [ ! -d "data-bin/tcr_2m" ]; then
    echo "Error: TCR dataset not found"
    echo "Run: python scripts/preprocess_tcr_data.py --input <pickle_file> --output data-bin/tcr_2m"
    exit 1
fi

# Set GPUs
export CUDA_VISIBLE_DEVICES=${DEVICES}

# Run training
python train.py \
    experiment=dplm/${CONFIG} \
    name=${CONFIG}_${DATE}

echo "Experiment completed: ${CONFIG}_${DATE}"



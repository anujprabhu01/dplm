#!/bin/bash
# Run model size scaling experiments on wolf server
# Usage: bash scripts/run_scaling_experiments.sh

set -e  # Exit on error

# Configuration
CONDA_ENV="dplm"
LOG_DIR="logs/scaling_experiments"
DATE=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "DPLM Model Size Scaling Experiments"
echo "Date: $DATE"
echo "========================================"

# Check if conda environment exists
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo -e "${RED}Error: Conda environment '${CONDA_ENV}' not found${NC}"
    echo "Please create it with: conda env create -f env.yml"
    exit 1
fi

# Activate conda environment
echo -e "${GREEN}Activating conda environment: ${CONDA_ENV}${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Check if data exists
if [ ! -d "data-bin/tcr_2m" ]; then
    echo -e "${RED}Error: TCR dataset not found at data-bin/tcr_2m${NC}"
    echo "Please run preprocessing first:"
    echo "  python scripts/preprocess_tcr_data.py \\"
    echo "    --input ~/data/tcr_dataset/tcr_repertoire_seqs.pkl \\"
    echo "    --output data-bin/tcr_2m \\"
    echo "    --max_sequences 2000000"
    exit 1
fi

# Create log directory
mkdir -p ${LOG_DIR}

# Function to run experiment
run_experiment() {
    local model_size=$1
    local experiment_name=$2
    local devices=$3
    local log_file="${LOG_DIR}/${experiment_name}_${DATE}.log"
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Starting: ${experiment_name}${NC}"
    echo -e "${GREEN}Model size: ${model_size}${NC}"
    echo -e "${GREEN}GPUs: ${devices}${NC}"
    echo -e "${GREEN}Log: ${log_file}${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Set GPU devices
    export CUDA_VISIBLE_DEVICES=${devices}
    
    # Run training
    python train.py \
        experiment=dplm/${experiment_name} \
        name=${experiment_name}_${DATE} \
        2>&1 | tee ${log_file}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${experiment_name} completed successfully${NC}"
    else
        echo -e "${RED}✗ ${experiment_name} failed${NC}"
        echo -e "${YELLOW}Check log: ${log_file}${NC}"
        return 1
    fi
}

# Run experiments sequentially (smallest to largest)
echo ""
echo -e "${YELLOW}Running experiments sequentially (smallest to largest)${NC}"
echo ""

# Experiment 1: 8M model (single GPU)
run_experiment "8M" "dplm_8m_tcr_2m" "0"

# Experiment 2: 35M model (single GPU)
run_experiment "35M" "dplm_35m_tcr_2m" "0"

# Experiment 3: 150M model (2 GPUs)
run_experiment "150M" "dplm_150m_tcr_2m" "0,1"

# Experiment 4: 650M model (4 GPUs)
run_experiment "650M" "dplm_650m_tcr_2m" "0,1,2,3"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved in: logs/train/runs/"
echo "Logs saved in: ${LOG_DIR}/"
echo ""
echo "Next steps:"
echo "1. Analyze results: python scripts/analyze_scaling_results.py"
echo "2. Generate sequences: bash scripts/generate_from_checkpoints.sh"
echo "3. Calculate pLDDT: bash scripts/calculate_plddt_scaling.sh"
echo ""



# DPLM Scaling Laws Scripts

This directory contains scripts for setting up and running scaling law experiments on DPLM.

## Quick Start

### 1. Create Dataset Subsets
```bash
# Create all standard subsets (1K, 10K, 100K, 500K, 1M)
python scripts/create_uniref50_subsets.py

# Or create custom sizes
python scripts/create_uniref50_subsets.py --sizes 1000 5000 10000
```

### 2. Run Smoke Test
```bash
# Quick test to verify everything works (5-10 minutes)
bash scripts/run_smoke_test.sh
```

### 3. Run Full Experiments
```bash
# See SCALING_LAWS_PLAN.md for detailed experimental protocols
```

---

## Scripts Overview

### `create_uniref50_subsets.py`
Creates multiple subsets of the UniRef50 dataset for systematic scaling experiments.

**Usage:**
```bash
python scripts/create_uniref50_subsets.py [OPTIONS]

Options:
  --output_dir DIR      Output directory (default: data-bin)
  --sizes SIZE [SIZE...]  Custom subset sizes
  --seed SEED           Random seed (default: 42)
  --download_full       Also save full dataset locally
```

**Examples:**
```bash
# Default: Create 1K, 10K, 100K, 500K, 1M subsets
python scripts/create_uniref50_subsets.py

# Custom sizes
python scripts/create_uniref50_subsets.py --sizes 2000 20000 200000

# Save full dataset too
python scripts/create_uniref50_subsets.py --download_full
```

**Output:**
```
data-bin/
  ├── uniref50_1k/         # 1,000 sequences
  ├── uniref50_10k/        # 10,000 sequences
  ├── uniref50_100k/       # 100,000 sequences
  ├── uniref50_500k/       # 500,000 sequences
  └── uniref50_1000k/      # 1,000,000 sequences
```

---

### `run_smoke_test.sh`
Runs a minimal training to verify setup works correctly.

**What it does:**
- Checks conda environment
- Creates tiny 1K dataset if needed
- Trains ESM2-35M for 100 steps
- Validates checkpointing and logging

**Usage:**
```bash
bash scripts/run_smoke_test.sh
```

**Expected Output:**
- Training completes in 5-10 minutes
- Checkpoints saved to `logs/train/runs/smoke_test_*/`
- TensorBoard logs available
- No errors during training

---

### `download_uniref50_hf.sh`
Downloads the full UniRef50 dataset from HuggingFace (~42M sequences).

**Usage:**
```bash
bash scripts/download_uniref50_hf.sh
```

**Note**: This downloads ~8GB of data. For scaling experiments, use `create_uniref50_subsets.py` instead to create only the sizes you need.

---

## Configuration Files

### Smoke Test Config
`configs/experiment/dplm/dplm_150m_smoke_test.yaml`

Optimized for quick testing:
- Model: ESM2-35M (lightweight)
- Dataset: 1K sequences
- Steps: 100
- Batch: 512 tokens
- Time: ~10 minutes

### Production Configs
- `dplm_150m.yaml` - 150M parameter model
- `dplm_650m.yaml` - 650M parameter model
- `dplm_3b.yaml` - 3B parameter model

---

## Workflow

### Phase 1: Setup & Smoke Test
```bash
# 1. Create tiny dataset
python scripts/create_uniref50_subsets.py --sizes 1000

# 2. Run smoke test
bash scripts/run_smoke_test.sh

# 3. Check results
tensorboard --logdir logs/
```

### Phase 2: Small-Scale Experiments
```bash
# 1. Create datasets
python scripts/create_uniref50_subsets.py --sizes 1000 10000 100000

# 2. Run experiments
python train.py \
    experiment=dplm/dplm_150m \
    name=dplm_150m_10k \
    datamodule.data_dir=data-bin/uniref50_10k \
    trainer.max_steps=5000
```

### Phase 3: Large-Scale Experiments
```bash
# 1. Create larger datasets
python scripts/create_uniref50_subsets.py --sizes 500000 1000000

# 2. Run with multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
    experiment=dplm/dplm_650m \
    name=dplm_650m_1m \
    datamodule.data_dir=data-bin/uniref50_1000k \
    trainer.devices=4 \
    trainer.max_steps=100000
```

---

## Troubleshooting

### Dataset Creation Issues

**Problem**: Slow download
```bash
# Solution: Pre-download to cache
export HF_HOME=/path/to/cache
python scripts/create_uniref50_subsets.py
```

**Problem**: Out of disk space
```bash
# Solution: Create only needed sizes
python scripts/create_uniref50_subsets.py --sizes 1000 10000
```

### Training Issues

**Problem**: CUDA out of memory
```bash
# Solution: Reduce batch size
python train.py \
    experiment=dplm/dplm_150m_smoke_test \
    datamodule.max_tokens=256
```

**Problem**: Training too slow
```bash
# Solution: Use mixed precision
python train.py \
    experiment=dplm/dplm_150m_smoke_test \
    trainer.precision=16
```

---

## Monitoring

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir logs/ --port 6006

# If on remote server, create SSH tunnel:
# Local machine:
ssh -L 6006:localhost:6006 adprabh1@10.173.14.250

# Then open: http://localhost:6006
```

### GPU Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or in tmux/screen
tmux new -s gpu_monitor
watch -n 1 nvidia-smi
# Ctrl+B, D to detach
```

---

## Tips

1. **Use tmux for long jobs**
   ```bash
   tmux new -s dplm_training
   python train.py experiment=...
   # Ctrl+B, D to detach
   # tmux attach -t dplm_training to reattach
   ```

2. **Save experiments systematically**
   ```bash
   # Use dated names
   name=dplm_150m_10k_$(date +%Y%m%d)
   ```

3. **Monitor from anywhere**
   ```bash
   # Use wandb for remote monitoring
   logger=wandb
   ```

4. **Checkpoint frequently**
   ```bash
   # For unstable environments
   callbacks.model_checkpoint.every_n_train_steps=100
   ```

---

## Additional Resources

- **Main Plan**: `../SCALING_LAWS_PLAN.md`
- **DPLM README**: `../README.md`
- **Config Examples**: `../configs/experiment/dplm/`

---

## Contact

For issues or questions about these scripts, refer to:
- DPLM paper: https://arxiv.org/abs/2402.18567
- DPLM repo: https://github.com/BytedProtein/dplm


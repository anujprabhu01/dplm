# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a fork of DPLM (Diffusion Protein Language Model) for conducting **scaling laws experiments**. The project systematically studies how model size, dataset size, and compute budget affect pretraining loss and generation quality for protein language models.

**Research Goals:**
- Quantify scaling relationships between model/data size and loss
- Identify compute-optimal training configurations
- Measure how generation quality (pLDDT scores) scales with model/data size

**Current Status (as of Dec 31, 2025):**
- ✅ Environment setup complete on wolf server (4x NVIDIA L40S GPUs, 48GB VRAM each)
- ✅ Conda environment created and all dependencies installed
- ✅ Forked DPLM repository to personal GitHub (anujprabhu01/dplm)
- ✅ Smoke test dataset created (1K sequences in `data-bin/uniref50_1k`)
- ✅ **Smoke test training completed successfully!**
  - Model: ESM2-35M (34M parameters)
  - Training: 100 steps completed in ~51 seconds
  - Validation perplexity decreased from 3.147 → 2.592 (model is learning!)
  - Checkpoint saved to: `logs/smoke_test_20251231_011053/checkpoints/`
- ✅ Generation smoke test scripts created (`scripts/test_generation_smoke.py`, `scripts/run_generation_smoke_test.sh`)
- ⏳ **NEXT**: Test generation from smoke test checkpoint
- ⏳ **NEXT**: Create larger dataset subsets (10K, 100K, 500K, 1M)
- ⏳ **NEXT**: Begin systematic scaling experiments

**Server Details:**
- Host: wolf.math.arizona.edu (10.173.14.250)
- User: adprabh1
- Working directory: `~/data/dplm` (symlinked to `/mnt/disk12/user/adprabh1/dplm`)
- Available storage: 23TB across 4 drives (disk12 is nearly empty with 5.5TB available)

## Immediate Next Steps

### 1. Test Generation Pipeline (NOW)
```bash
# On wolf server
cd ~/data/dplm
conda activate dplm

# Run generation smoke test
bash scripts/run_generation_smoke_test.sh

# This will:
# - Find your latest checkpoint automatically
# - Load the model
# - Generate 5 sequences of length 100
# - Save results to generation-results/smoke_test/
```

**Expected outcome:** Script completes without errors and generates 5 protein sequences. Note: Sequences may look random since the model only trained for 100 steps - this is expected! The goal is to verify the generation pipeline works.

### 2. Create Larger Datasets
```bash
# Create datasets for scaling experiments
python scripts/create_uniref50_subsets.py --sizes 10000 100000 500000 1000000

# This will create:
# - data-bin/uniref50_10k/    (10K sequences)
# - data-bin/uniref50_100k/   (100K sequences)
# - data-bin/uniref50_500k/   (500K sequences)
# - data-bin/uniref50_1000k/  (1M sequences)
```

### 3. Run First Real Scaling Experiment
```bash
# Example: 150M model on 10K dataset
python train.py \
    experiment=dplm/dplm_150m \
    name=dplm_150m_10k_$(date +%Y%m%d) \
    datamodule.data_dir=data-bin/uniref50_10k \
    trainer.max_steps=5000 \
    trainer.devices=1
```

## Key Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n dplm python=3.9 pip
conda activate dplm

# Install dependencies
cd dplm
bash scripts/install.sh
```

### Dataset Creation
```bash
# Create subsets for scaling experiments
python scripts/create_uniref50_subsets.py --sizes 1000 10000 100000

# Download full UniRef50 (~8GB, 42M sequences)
bash scripts/download_uniref50_hf.sh
```

### Training

**Smoke Test (5-10 minutes):**
```bash
bash scripts/run_smoke_test.sh
```

**Basic Training:**
```bash
python train.py \
    experiment=dplm/dplm_150m \
    name=experiment_name \
    datamodule.data_dir=data-bin/uniref50_10k \
    trainer.max_steps=5000
```

**Multi-GPU Training:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
    experiment=dplm/dplm_650m \
    name=dplm_650m_1m \
    datamodule.max_tokens=8192 \
    trainer.accumulate_grad_batches=16 \
    trainer.devices=4
```

**Resume from Checkpoint:**
```bash
python train.py \
    experiment=dplm/dplm_150m \
    ckpt_path=logs/train/runs/experiment_name/checkpoints/last.ckpt
```

### Testing
```bash
python test.py \
    experiment_path=path/to/experiment \
    ckpt_path=best.ckpt \
    data_split=test \
    mode=predict
```

### Generation

**DPLM (sequence-only):**
```bash
python generate_dplm.py \
    --model_name airkingbd/dplm_650m \
    --seq_lens 100 200 300 \
    --num_seqs 50 \
    --saveto generation-results/dplm_650m
```

**DPLM-2 (sequence + structure co-generation):**
```bash
python generate_dplm2.py \
    --model_name airkingbd/dplm2_650m \
    --task co_generation \
    --sampling_strategy annealing@2.0:0.1 \
    --num_seqs 50 \
    --max_iter 500 \
    --seq_lens 100 200 300 \
    --saveto generation-results/dplm2_650m
```

**Forward Folding (sequence → structure):**
```bash
python generate_dplm2.py \
    --model_name airkingbd/dplm2_650m \
    --task folding \
    --input_fasta_path data-bin/sequences.fasta \
    --max_iter 100 \
    --unmasking_strategy deterministic \
    --sampling_strategy argmax
```

**Inverse Folding (structure → sequence):**
```bash
# First tokenize structures
python src/byprot/utils/protein/tokenize_pdb.py \
    --input_pdb_folder /path/to/pdbs \
    --output_dir tokenized_output

# Then generate sequences
python generate_dplm2.py \
    --model_name airkingbd/dplm2_650m \
    --task inverse_folding \
    --input_fasta_path tokenized_output/struct.fasta
```

### Evaluation
```bash
# Calculate pLDDT using ESMFold
bash analysis/plddt_calculate.sh generation-results/dplm_650m

# DPLM-2 evaluation (RMSD, TM-score)
python src/byprot/utils/protein/evaluator_dplm2.py -cn unconditional_codesign \
    inference.input_fasta_dir=generation-results/dplm2_650m
```

### Monitoring
```bash
# TensorBoard (local)
tensorboard --logdir logs/ --port 6006

# TensorBoard (remote server with SSH tunnel)
# On server:
tensorboard --logdir logs/ --port 6006
# On local machine:
ssh -L 6006:localhost:6006 user@server
# Then open: http://localhost:6006

# GPU monitoring
watch -n 1 nvidia-smi
```

## Architecture Overview

### Configuration System (Hydra-based)
- **Main entry**: `configs/config.yaml` composes all sub-configs
- **Experiments**: `configs/experiment/dplm/` contains model-specific configs
  - `dplm_150m.yaml`, `dplm_650m.yaml`, `dplm_3b.yaml`
  - `dplm_150m_smoke_test.yaml` for quick testing
- **Override syntax**: `python train.py experiment=dplm/dplm_150m datamodule.max_tokens=4096`
- **Config resolution**: Hydra merges defaults → experiment config → CLI overrides

### Training Pipeline
1. **Entry point**: `train.py` uses `@hydra.main` decorator
2. **Root setup**: `pyrootutils.setup_root()` sets `PROJECT_ROOT`, loads `.env`
3. **Pipeline orchestration**: `byprot.training_pipeline.train(config)`
   - Instantiates datamodule, task, model via **registry pattern**
   - Sets up PyTorch Lightning Trainer with callbacks/loggers
   - Runs `Trainer.fit(pl_module, datamodule)`
4. **Checkpointing**: Saves to `logs/train/runs/{experiment_name}/checkpoints/`
   - `best.ckpt` (based on validation metric)
   - `last.ckpt` (most recent)

### Model Architecture
- **Base models**: Located in `src/byprot/models/`
  - `dplm/`: Sequence-only diffusion model (33 tokens: 20 AA + special tokens)
  - `dplm2/`: Multimodal (sequence + structure tokens, vocab size: 33 + 8192 + 4)
  - `dplm2/dplm2_bit.py`: Bit-based structure modeling variant
- **Backbone**: Uses ESM2 transformer (`facebook/esm2_*`) with modifications
- **Registry**: Models registered via `@register_model()` decorator
- **Loading**: `Model.from_pretrained("airkingbd/dplm_650m")` loads from HuggingFace

### Task System (PyTorch Lightning)
- **Base class**: `TaskLitModule` extends `LightningModule`
- **Task types**: `DPLMTrainingTask`, `DPLM2TrainingTask` in `src/byprot/tasks/lm/`
- **Responsibilities**:
  - Training/validation/test step logic
  - Loss computation (diffusion-specific q_sample + cross-entropy)
  - Optimizer/scheduler configuration
  - Metrics logging (perplexity, accuracy)

### Data Handling
- **DataModules**: `src/byprot/datamodules/`
  - `uniref50.py`: Sequence pretraining data
  - `cath_datamodule.py`: Structure data for inverse folding
  - `uniref50_hf.py`: HuggingFace Datasets backend
- **Batching**: `MaxTokensBatchSampler` groups sequences by length for efficiency
- **Tokenization**: Custom tokenizers for amino acids + structure tokens

### Diffusion Mechanism
- **Forward diffusion**: `q_sample()` progressively masks tokens based on timestep `t`
- **Training objective**: Predict original tokens from partially masked inputs
- **Generation**: Iterative denoising from fully masked input
  - Unmasking strategies: `stochastic`, `deterministic`
  - Sampling strategies: `gumbel_argmax`, `vanilla`, `annealing@T_start:T_end`
- **Coupled training**: Uses two timesteps (t1, t2) for improved conditioning

### Generation Workflow
```
Load model → Initialize masked tokens → Iterative denoising (max_iter steps)
  ├─ Forward pass: model.forward_decoder(masked_tokens)
  ├─ Sample tokens: Apply sampling strategy (gumbel/vanilla/argmax)
  ├─ Unmask positions: Follow unmasking schedule
  └─ Repeat until convergence
→ Decode tokens → Save FASTA/PDB
```

### Key Abstractions
1. **Registry Pattern**: All models/tasks/datamodules registered centrally
   - `@register_model()`, `@register_task()`, `@register_datamodule()`
   - Enables config-driven component selection
2. **Hydra Composition**: Flexible experiment configuration via YAML composition
3. **PyTorch Lightning**: Handles distributed training (DDP/FSDP), checkpointing, logging
4. **Structure Tokenizer**: Converts 3D structures ↔ discrete tokens (DPLM-2)
5. **LoRA Support**: Efficient fine-tuning with low-rank adapters

## Project-Specific Details

### Scaling Laws Experiments
This fork includes additional scripts for systematic scaling studies:
- `scripts/create_uniref50_subsets.py`: Creates datasets of varying sizes (1K → 1M sequences)
- `scripts/run_smoke_test.sh`: Quick validation of setup
- `SCALING_LAWS_PLAN.md`: Detailed experimental protocol
- `NEXT_STEPS.md`: Step-by-step execution guide

**Standard Dataset Sizes:**
- Smoke test: 1K sequences
- Small-scale: 10K, 100K
- Medium-scale: 500K
- Large-scale: 1M, 5M

### Effective Batch Size Calculation
Total tokens per update = `max_tokens × devices × accumulate_grad_batches`

**Examples:**
- DPLM pretraining: ~1M tokens/batch (8192 × 8 GPUs × 16 accum)
- DPLM-2 pretraining: ~64K tokens/batch (8192 × 8 GPUs × 1 accum)
- Smoke test: ~4K tokens/batch (512 × 1 GPU × 8 accum)

### Model Sizes
| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| DPLM-150M | 150M | `airkingbd/dplm_150m` |
| DPLM-650M | 650M | `airkingbd/dplm_650m` |
| DPLM-3B | 3B | `airkingbd/dplm_3b` |
| DPLM2-150M | 150M | `airkingbd/dplm2_150m` |
| DPLM2-650M | 650M | `airkingbd/dplm2_650m` |
| DPLM2-3B | 3B | `airkingbd/dplm2_3b` |
| DPLM2-Bit-650M | 650M | `airkingbd/dplm2_bit_650m` |

### Training Strategies

**DPLM (sequence-only):**
- Pretrain on UniRef50 (42M sequences)
- ~100K steps, 1M tokens/batch
- Standard cross-entropy + diffusion objective

**DPLM-2 (multimodal):**
- Initialize from pretrained DPLM checkpoint
- LoRA fine-tuning to preserve evolutionary knowledge
- Train on PDB + SwissProt (experimental + predicted structures)
- ~100K steps, 64K tokens/batch

**DPLM-2 Bit:**
- Bit-based structure token prediction (instead of index-based)
- Better structure modeling performance
- Same training protocol as DPLM-2

### Common Configuration Overrides

**Reduce memory usage:**
```bash
datamodule.max_tokens=2048          # Smaller batch size
model.gradient_ckpt=true            # Gradient checkpointing
trainer.precision=16                # Mixed precision (FP16)
```

**Speed up training:**
```bash
trainer.devices=4                   # Multi-GPU
trainer.accumulate_grad_batches=4   # Gradient accumulation
datamodule.num_workers=8            # Parallel data loading
```

**Debugging:**
```bash
trainer.max_steps=100               # Limit steps
trainer.fast_dev_run=true           # Quick sanity check
trainer.limit_train_batches=10      # Limit batches per epoch
```

## Important Project-Specific Notes

### OpenFold Installation (Known Issue)
The vendored OpenFold package (`vendor/openfold/`) **fails to compile** due to CUDA kernel compilation errors. This is a known issue and has been documented.

**Good news:** OpenFold is **optional** for your scaling laws experiments! It's only needed for:
- Structure prediction evaluation (alternative to ESMFold)
- DPLM-2 structure-conditioned tasks

For your current work (DPLM sequence generation and scaling laws), you **do not need OpenFold**. All core DPLM functionality works without it.

If you need structure evaluation later, use ESMFold via the provided scripts in `analysis/`.

### Smoke Test Results
Your smoke test on Dec 31, 2025 was successful:
- **Checkpoint location**: `~/data/dplm/logs/smoke_test_20251231_011053/checkpoints/`
- **Model**: ESM2-35M (34M parameters)
- **Training time**: ~51 seconds for 100 steps
- **Learning confirmed**: Validation perplexity decreased from 3.147 → 2.592
- **Memory usage**: Fits comfortably on single L40S GPU with FP16 precision

### Generation Scripts Available
Two new scripts were created for testing generation:
1. `scripts/test_generation_smoke.py` - Python script for unconditional generation
2. `scripts/run_generation_smoke_test.sh` - Bash wrapper that auto-finds checkpoint

These have **not been tested yet** - they are ready for you to run.

## Important Notes

### File Locations
- **Training logs**: `logs/train/runs/{experiment_name}/`
- **Checkpoints**: `logs/train/runs/{experiment_name}/checkpoints/`
- **Generated outputs**: `generation-results/`
- **Datasets**: `data-bin/`
- **Analysis scripts**: `analysis/` (Jupyter notebooks for plotting)

### Vendor Code
- `vendor/openfold/`: Modified OpenFold implementation for structure prediction
- Generally read-only; modifications should be rare

### Checkpoint Loading
When loading checkpoints in code:
```python
# From HuggingFace
from byprot.models.dplm import DiffusionProteinLanguageModel as DPLM
model = DPLM.from_pretrained("airkingbd/dplm_650m")

# From local checkpoint
model = DPLM.from_pretrained("/path/to/checkpoint.ckpt")
```

### Configuration Inheritance
Experiment configs can inherit from base configs:
```yaml
# In experiment config
defaults:
  - override /datamodule: uniref50
  - override /model: dplm_650m

# This merges configs/datamodule/uniref50.yaml and configs/model/dplm_650m.yaml
```

### Sampling Strategies for Generation
- `argmax`: Deterministic, best for structure prediction
- `gumbel_argmax`: Adds controlled noise for diversity
- `vanilla`: Stochastic sampling with temperature
- `annealing@T_high:T_low`: Start diverse (T_high), end focused (T_low)

### Evaluation Metrics
- **pLDDT**: Predicted local distance difference test (structure confidence, 0-100)
  - Good: >70
  - High quality: >90
- **RMSD**: Root-mean-square deviation (Å, lower is better)
- **TM-score**: Template modeling score (0-1, higher is better, >0.5 is good)
- **scTM**: Self-consistency TM-score (sequence → structure → sequence fidelity)

### Performance Tips
1. Use `tmux` or `screen` for long-running jobs on remote servers
2. Monitor with TensorBoard during training
3. Save experiment configs alongside results for reproducibility
4. Use dated experiment names: `dplm_150m_10k_20251231`
5. Check `nvidia-smi` before launching to verify GPU availability

## Scaling Experiments Plan

Based on your `SCALING_LAWS_PLAN.md`, here's the recommended experimental sequence:

### Phase 1: ✅ Smoke Test (COMPLETED)
- Model: ESM2-35M (34M params)
- Dataset: 1K sequences
- Steps: 100
- Status: **DONE** ✅

### Phase 2A: Dataset Size Scaling (RECOMMENDED NEXT)
Fix model size at 150M, vary dataset size:

| Experiment | Model | Dataset | Steps | Priority | Estimated Time |
|-----------|-------|---------|-------|----------|----------------|
| dplm_150m_1k | 150M | 1K | 1,000 | HIGH | ~30 min |
| dplm_150m_10k | 150M | 10K | 5,000 | HIGH | ~2 hrs |
| dplm_150m_100k | 150M | 100K | 20,000 | MEDIUM | ~8 hrs |
| dplm_150m_500k | 150M | 500K | 50,000 | LOW | 1-2 days |

### Phase 2B: Model Size Scaling
Fix dataset at 100K, vary model size:

| Experiment | Model | Dataset | Steps | Priority | Estimated Time |
|-----------|-------|---------|-------|----------|----------------|
| dplm_35m_100k | 35M | 100K | 20,000 | HIGH | ~4 hrs |
| dplm_150m_100k | 150M | 100K | 20,000 | HIGH | ~8 hrs |
| dplm_650m_100k | 650M | 100K | 20,000 | LOW | 2-3 days |

### Phase 3: Full Scaling (If Compute Allows)
- Larger datasets (1M, 5M sequences)
- Larger models (650M, 3B params)
- Multi-GPU training required

### Key Metrics to Log
For each experiment, record:
- Training/validation loss curves
- Perplexity
- Tokens/second throughput
- GPU memory usage
- Total training time
- Checkpoint paths
- Generated sequence quality (pLDDT scores)

Refer to `SCALING_LAWS_PLAN.md` and `NEXT_STEPS.md` for complete details.

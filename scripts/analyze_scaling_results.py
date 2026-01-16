#!/usr/bin/env python3
"""
Analyze model size scaling experiment results.

This script extracts training metrics from TensorBoard logs and creates
scaling law plots.

Usage:
    python scripts/analyze_scaling_results.py \
        --log_dir logs/train/runs \
        --output results/scaling_analysis.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tensorboard.backend.event_processing import event_accumulator
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Model parameter counts
MODEL_PARAMS = {
    'dplm_8m_tcr_2m': 8_000_000,
    'dplm_35m_tcr_2m': 35_000_000,
    'dplm_150m_tcr_2m': 150_000_000,
    'dplm_650m_tcr_2m': 650_000_000
}


def extract_tensorboard_data(log_dir: Path, run_name: str) -> Dict[str, np.ndarray]:
    """Extract training metrics from TensorBoard logs."""
    logger.info(f"Extracting data from {run_name}")
    
    # Find tensorboard log directory
    tb_dir = log_dir / run_name / 'tensorboard'
    
    if not tb_dir.exists():
        logger.warning(f"TensorBoard directory not found: {tb_dir}")
        return {}
    
    # Load event files
    ea = event_accumulator.EventAccumulator(str(tb_dir))
    ea.Reload()
    
    data = {}
    
    # Extract scalar metrics
    for tag in ['train/loss', 'val/loss', 'train/perplexity', 'val/perplexity']:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = {'steps': np.array(steps), 'values': np.array(values)}
    
    return data


def compute_final_metrics(data: Dict[str, np.ndarray], 
                          window_size: int = 1000) -> Dict[str, float]:
    """Compute final metrics (averaged over last window_size steps)."""
    metrics = {}
    
    for key, values in data.items():
        if 'values' in values and len(values['values']) > 0:
            # Average over last window_size values
            final_values = values['values'][-window_size:]
            metrics[key] = float(np.mean(final_values))
            metrics[f'{key}_std'] = float(np.std(final_values))
    
    return metrics


def collect_all_results(log_dir: Path) -> pd.DataFrame:
    """Collect results from all experiments."""
    logger.info("Collecting results from all experiments")
    
    results = []
    
    for model_name, params in MODEL_PARAMS.items():
        # Find all runs for this model
        matching_runs = list(log_dir.glob(f'{model_name}_*'))
        
        if not matching_runs:
            logger.warning(f"No runs found for {model_name}")
            continue
        
        # Use most recent run
        run_dir = max(matching_runs, key=lambda p: p.stat().st_mtime)
        run_name = run_dir.name
        
        logger.info(f"Processing {run_name}")
        
        # Extract data
        data = extract_tensorboard_data(log_dir, run_name)
        
        if not data:
            logger.warning(f"No data extracted for {run_name}")
            continue
        
        # Compute metrics
        metrics = compute_final_metrics(data)
        
        # Add to results
        result = {
            'model_name': model_name,
            'run_name': run_name,
            'parameters': params,
            **metrics
        }
        
        results.append(result)
    
    df = pd.DataFrame(results)
    df = df.sort_values('parameters')
    
    return df


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit power law: y = a * x^(-b) + c
    Returns (a, b, c)
    """
    from scipy.optimize import curve_fit
    
    def power_law(x, a, b, c):
        return a * np.power(x, -b) + c
    
    try:
        params, _ = curve_fit(power_law, x, y, p0=[1.0, 0.1, 0.0])
        return tuple(params)
    except Exception as e:
        logger.error(f"Power law fit failed: {e}")
        return (np.nan, np.nan, np.nan)


def plot_scaling_curves(df: pd.DataFrame, output_dir: Path):
    """Create scaling law plots."""
    logger.info("Creating scaling law plots")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Loss vs Model Size (log-log)
    plt.figure(figsize=(10, 6))
    
    if 'train/loss' in df.columns:
        plt.loglog(df['parameters'], df['train/loss'], 'o-', label='Train Loss', markersize=8)
    
    if 'val/loss' in df.columns:
        plt.loglog(df['parameters'], df['val/loss'], 's-', label='Val Loss', markersize=8)
    
    # Fit power law to train loss
    if 'train/loss' in df.columns and len(df) >= 3:
        a, b, c = fit_power_law(df['parameters'].values, df['train/loss'].values)
        
        if not np.isnan(b):
            x_fit = np.logspace(np.log10(df['parameters'].min()), 
                               np.log10(df['parameters'].max()), 100)
            y_fit = a * np.power(x_fit, -b) + c
            plt.loglog(x_fit, y_fit, '--', alpha=0.5, label=f'Power Law (b={b:.4f})')
    
    plt.xlabel('Model Size (Parameters)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('DPLM Scaling Law: Loss vs Model Size\n(Fixed 2M TCR Sequences)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_law_loss.png', dpi=300)
    plt.close()
    logger.info(f"Saved plot: {output_dir / 'scaling_law_loss.png'}")
    
    # Plot 2: Perplexity vs Model Size
    plt.figure(figsize=(10, 6))
    
    if 'train/perplexity' in df.columns:
        plt.loglog(df['parameters'], df['train/perplexity'], 'o-', 
                  label='Train Perplexity', markersize=8)
    
    if 'val/perplexity' in df.columns:
        plt.loglog(df['parameters'], df['val/perplexity'], 's-', 
                  label='Val Perplexity', markersize=8)
    
    plt.xlabel('Model Size (Parameters)', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('DPLM Scaling: Perplexity vs Model Size\n(Fixed 2M TCR Sequences)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_law_perplexity.png', dpi=300)
    plt.close()
    logger.info(f"Saved plot: {output_dir / 'scaling_law_perplexity.png'}")
    
    # Plot 3: Loss improvement per parameter
    if 'train/loss' in df.columns and len(df) > 1:
        plt.figure(figsize=(10, 6))
        
        # Compute relative improvement
        baseline_loss = df['train/loss'].iloc[0]  # Smallest model
        relative_improvement = (baseline_loss - df['train/loss']) / baseline_loss * 100
        params_ratio = df['parameters'] / df['parameters'].iloc[0]
        
        plt.semilogx(params_ratio, relative_improvement, 'o-', markersize=8)
        plt.xlabel('Model Size Ratio (relative to 8M)', fontsize=12)
        plt.ylabel('Loss Improvement (%)', fontsize=12)
        plt.title('Efficiency: Loss Improvement vs Model Size Increase', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'scaling_efficiency.png', dpi=300)
        plt.close()
        logger.info(f"Saved plot: {output_dir / 'scaling_efficiency.png'}")


def save_summary(df: pd.DataFrame, output_path: Path):
    """Save summary statistics and power law fits."""
    logger.info(f"Saving summary to {output_path}")
    
    summary = {
        'experiment': 'Model Size Scaling (Fixed 2M TCR Data)',
        'num_models': len(df),
        'model_sizes': df['parameters'].tolist(),
        'results': df.to_dict('records')
    }
    
    # Add power law fits
    if 'train/loss' in df.columns and len(df) >= 3:
        a, b, c = fit_power_law(df['parameters'].values, df['train/loss'].values)
        summary['power_law_fit'] = {
            'equation': 'Loss = a * N^(-b) + c',
            'a': float(a),
            'b': float(b),
            'c': float(c),
            'interpretation': f'Loss scales as N^(-{b:.4f})'
        }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze DPLM scaling results')
    parser.add_argument('--log_dir', type=str, default='logs/train/runs',
                       help='Directory containing experiment logs')
    parser.add_argument('--output_dir', type=str, default='results/scaling_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results
    df = collect_all_results(log_dir)
    
    if df.empty:
        logger.error("No results found!")
        return
    
    # Save CSV
    csv_path = output_dir / 'scaling_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("DPLM Model Size Scaling Results")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60 + "\n")
    
    # Create plots
    plot_scaling_curves(df, output_dir)
    
    # Save detailed summary
    save_summary(df, output_dir / 'summary.json')
    
    logger.info("Analysis complete!")
    logger.info(f"Results saved in: {output_dir}")


if __name__ == '__main__':
    main()



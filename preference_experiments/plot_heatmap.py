import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import matplotlib.patheffects as path_effects
from utils_logging import get_logger

def make_heatmap_matrix(exp_dir, metric='self_preference_rate', logger=None):
    """
    Generate a heatmap matrix of self-preference or detection accuracy scores for a given experiment folder.
    Args:
        exp_dir (str): Path to the experiment output folder (containing metadata.json, model subfolders, etc.)
        metric (str): 'self_preference_rate' or 'detection_accuracy'
        logger (logging.Logger): Logger for output
    """
    # Load metadata
    metadata_path = os.path.join(exp_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {exp_dir}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    dataset = metadata['dataset']
    N = metadata['N']
    models = metadata['models']
    experiment_id = metadata['experiment_id']

    heatmap_dir = os.path.join(exp_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)

    # Sort models alphabetically
    evaluators = sorted(models)
    all_models = sorted(models)

    # Build matrix
    matrix = pd.DataFrame(index=evaluators, columns=all_models, dtype=float)
    for evaluator in evaluators:
        model_dir = os.path.join(exp_dir, evaluator)
        if metric == 'self_preference_rate':
            csv_file = os.path.join(model_dir, f"{evaluator}_comparison_results_self_prefer_rate_simple.csv")
        elif metric == 'detection_accuracy':
            csv_file = os.path.join(model_dir, f"{evaluator}_comparison_results_detect_accuracy_simple.csv")
        else:
            raise ValueError('Unknown metric')
        if not os.path.exists(csv_file):
            if logger:
                logger.warning(f"{csv_file} not found, skipping {evaluator}")
            else:
                print(f"Warning: {csv_file} not found, skipping {evaluator}")
            continue
        df = pd.read_csv(csv_file, index_col=0, header=0)
        for model in all_models:
            if model == evaluator:
                matrix.loc[evaluator, model] = np.nan
            elif model in df.index:
                matrix.loc[evaluator, model] = df.loc[model].values[0]
            elif model in df.columns:
                matrix.loc[evaluator, model] = df[model].values[0]
            else:
                matrix.loc[evaluator, model] = np.nan

    # Save matrix as CSV
    matrix_csv = os.path.join(heatmap_dir, f"{metric}_heatmap_{experiment_id}.csv")
    matrix.to_csv(matrix_csv)

    # Plot heatmap
    plt.figure(figsize=(max(8, len(all_models)), max(6, len(evaluators))))
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"]
    cmap = plt.get_cmap('hot')
    im = plt.imshow(matrix.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label=metric.replace('_', ' ').title())
    plt.xticks(ticks=np.arange(len(all_models)), labels=all_models, rotation=45, ha='right', fontsize=12)
    plt.yticks(ticks=np.arange(len(evaluators)), labels=evaluators, fontsize=12)
    plt.title(f"{metric.replace('_', ' ').title()} Heatmap ({dataset}, N={N})", fontsize=16)
    # Annotate cells with bold, outlined text for readability
    for i in range(len(evaluators)):
        for j in range(len(all_models)):
            if i == j:
                continue
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                color = 'white'
                txt = plt.text(
                    j, i, f"{val:.2f}", ha='center', va='center',
                    color=color, fontsize=14,
                )
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=1, foreground='black'),
                    path_effects.Normal()
                ])
    plt.tight_layout()
    matrix_pdf = os.path.join(heatmap_dir, f"{metric}_heatmap_{experiment_id}.pdf")
    plt.savefig(matrix_pdf, bbox_inches='tight')
    plt.close()
    if logger:
        logger.info(f"Saved heatmap matrix to {matrix_csv} and {matrix_pdf}")
    else:
        print(f"Saved heatmap matrix to {matrix_csv} and {matrix_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate heatmap matrix for self-preference or detection accuracy for a given experiment.')
    parser.add_argument('--exp_dir', type=str, required=True, help='Experiment output folder (containing metadata.json, model subfolders, etc.)')
    parser.add_argument('--metric', type=str, default='self_preference_rate', choices=['self_preference_rate', 'detection_accuracy'], help='Metric to plot')
    args = parser.parse_args()
    # Set up logger
    try:
        with open(os.path.join(args.exp_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        log_level = metadata.get('log_level', 'INFO')
    except Exception:
        log_level = 'INFO'
    from utils_logging import get_logger
    logger = get_logger(args.exp_dir, log_level=log_level)
    make_heatmap_matrix(args.exp_dir, args.metric, logger=logger) 
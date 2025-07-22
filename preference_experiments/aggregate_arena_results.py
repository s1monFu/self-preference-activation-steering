import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_arena_results(exp_dir):
    """Load all arena result files from the experiment directory."""
    results = {}
    pattern = os.path.join(exp_dir, "*_arena_preference_results.json")
    
    for file_path in glob.glob(pattern):
        model_name = os.path.basename(file_path).split("_arena_preference_results")[0]
        with open(file_path, 'r') as f:
            results[model_name] = json.load(f)
    
    return results

def calculate_inverse_self_preference(data, min_count=25):
    """
    Calculate P(pick='self' | won=0) and bias for each self/other pair.
    Returns a dict with 'prob_self_given_wrong', 'bias', and other stats.
    """
    grouped = defaultdict(list)
    for entry in data:
        grouped[entry['reference_model']].append(entry)
    results = {}
    for other_model, entries in grouped.items():
        won0_entries = [e for e in entries if e.get('won', 0) == 0]
        if len(won0_entries) < min_count:
            results[other_model] = None
            continue
        n_self = sum(e['self_pick'] for e in won0_entries )#if e['pick'] == 'self')
        n_other = sum(e['other_pick'] for e in won0_entries )#if e['pick'] == 'other')
        total = len(won0_entries)
        prob_self = n_self / total if total > 0 else 0
        prob_other = n_other / total if total > 0 else 0
        bias = prob_self - prob_other
        results[other_model] = {
            'prob_self_given_wrong': prob_self,
            'prob_other_given_wrong': prob_other,
            'bias': bias,
            'n_self': n_self,
            'n_other': n_other,
            'total': total
        }
    return results

def create_heatmap(all_results, exp_dir, min_count=25):
    """
    Create a heatmap showing P(pick='self' | won=0) for each self/other pair, with bias in superscript if > 0.
    Only self models are rows, columns are labeled by other models, and squares are larger.
    """
    # Only use self models that are present in the results
    self_models = sorted(list(all_results.keys()))
    # Collect all unique other models from the data
    other_models = set()
    for model_data in all_results.values():
        other_models.update(model_data.keys())
    other_models = sorted(list(other_models))
    # Create matrix
    matrix = np.full((len(self_models), len(other_models)), np.nan)
    bias_matrix = np.full((len(self_models), len(other_models)), np.nan)
    for i, self_model in enumerate(self_models):
        for j, other_model in enumerate(other_models):
            if self_model == other_model:
                continue  # Skip diagonal
            if self_model in all_results and other_model in all_results[self_model]:
                result = all_results[self_model][other_model]
                if result is not None:
                    matrix[i, j] = result['prob_self_given_wrong']
                    bias_matrix[i, j] = result['bias']
    # Remove columns (other models) with all NaN
    valid_cols = [j for j in range(len(other_models)) if not np.all(np.isnan(matrix[:, j]))]
    filtered_other_models = [other_models[j] for j in valid_cols]
    filtered_matrix = matrix[:, valid_cols]
    filtered_bias_matrix = bias_matrix[:, valid_cols]
    filtered_mask = np.isnan(filtered_matrix)
    plt.figure(figsize=(2 + len(filtered_other_models) * 2, 2 + len(self_models) * 2))
    ax = sns.heatmap(filtered_matrix, annot=False, fmt='.3f', cmap='RdYlBu_r', center=0.5, mask=filtered_mask,
                cbar_kws={'label': "P(pick='self' | won=0)"}, square=True,
                xticklabels=filtered_other_models, yticklabels=self_models, linewidths=2, linecolor='white')
    # Annotate with value and bias in superscript if bias > 0
    for i in range(len(self_models)):
        for idx_j, j in enumerate(valid_cols):
            if not np.isnan(matrix[i, j]):
                val = f"{matrix[i, j]:.3f}"
                bias = bias_matrix[i, j]
                if bias > 0:
                    val += f"$^{{({bias:.2f})}}$"
                ax.text(idx_j + 0.5, i + 0.5, val, ha='center', va='center', fontsize=18, color='black')
    plt.title(f"P(pick='self' | won=0) Heatmap\n(Min count: {min_count}, bias in superscript if > 0)", fontsize=18)
    plt.xlabel('Other Model', fontsize=16)
    plt.ylabel('Self Model', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    output_path = os.path.join(exp_dir, f'arena_inverse_self_preference_heatmap_min{min_count}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {output_path}")
    df = pd.DataFrame(filtered_matrix, index=self_models, columns=filtered_other_models)
    csv_path = os.path.join(exp_dir, f'arena_inverse_self_preference_matrix_min{min_count}.csv')
    df.to_csv(csv_path)
    print(f"Matrix data saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate arena experiment results and create heatmap.")
    parser.add_argument("exp_dir", help="Path to arena experiment directory")
    parser.add_argument("--min_count", type=int, default=25, 
                       help="Minimum number of responses required for a pair (default: 25)")
    args = parser.parse_args()
    print(f"Loading results from: {args.exp_dir}")
    all_results = load_arena_results(args.exp_dir)
    if not all_results:
        print("No arena result files found!")
        return
    print(f"Found results for {len(all_results)} models")
    aggregated_results = {}
    for model_name, model_data in all_results.items():
        print(f"Processing {model_name}...")
        aggregated_results[model_name] = calculate_inverse_self_preference(
            model_data, args.min_count
        )
    print("Creating heatmap...")
    create_heatmap(aggregated_results, args.exp_dir, args.min_count)
    print("\nSummary:")
    for self_model, other_results in aggregated_results.items():
        valid_pairs = sum(1 for result in other_results.values() if result is not None)
        total_pairs = len(other_results)
        print(f"{self_model}: {valid_pairs}/{total_pairs} valid pairs")
    print("Done!")

if __name__ == "__main__":
    main() 
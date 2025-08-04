import os
import glob
import json
import random
import argparse
from collections import defaultdict
import re

def merge_summaries_for_all_models(dataset):
    summary_dir = f"summaries/{dataset}"
    pattern = os.path.join(summary_dir, f"{dataset}_train_*_responses*.json")
    files = glob.glob(pattern)
    # Find all models by parsing filenames
    model_files = defaultdict(list)
    model_pattern = re.compile(rf"{dataset}_train_(.+?)_responses.*\.json$")
    for file in files:
        match = model_pattern.search(os.path.basename(file))
        if match:
            model = match.group(1)
            model_files[model].append(file)
    for model, model_file_list in model_files.items():
        all_summaries = defaultdict(list)
        for file in model_file_list:
            with open(file, "r") as f:
                try:
                    summaries = json.load(f)
                    for k, v in summaries.items():
                        all_summaries[k].append(v)
                except Exception as e:
                    print(f"Warning: Could not read {file}: {e}")
        merged = {k: random.choice([v for v in vs if v is not None and v != ""]) for k, vs in all_summaries.items() if any(v is not None and v != "" for v in vs)}
        merged_file = os.path.join(summary_dir, f"{dataset}_train_{model}_responses_merged.json")
        with open(merged_file, "w") as f:
            json.dump(merged, f, indent=2)
        # Move old files to messy/
        messy_dir = os.path.join(summary_dir, "messy")
        os.makedirs(messy_dir, exist_ok=True)
        for file in model_file_list:
            os.rename(file, os.path.join(messy_dir, os.path.basename(file)))
        print(f"Merged {len(model_file_list)} files for {model} in {dataset} into {merged_file} ({len(merged)} unique keys).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all summary files for all models in a dataset into merged files.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., cnn or xsum)")
    args = parser.parse_args()
    merge_summaries_for_all_models(args.dataset) 
from models import get_code, get_summary, code_datasets
from tqdm import tqdm
import os
import json
import argparse

def load_from_json(file_name) -> dict:
    """Load a dictionary from a JSON file."""
    with open(file_name, "r") as f:
        return json.load(f)

def load_sources(dataset, extras=False):
    """
    Load articles for a given dataset and set of sources.
    Returns (sources, keys).
    """
    data_type = "code" if dataset in code_datasets else "sources"
    articles = load_from_json(f"{data_type}/{dataset}_train_{data_type}{'_extra' if extras else ''}.json")
    keys = list(articles.keys())
    return articles, keys

def process_dataset(dataset, model, N, overwrite=False, extras=False):
    merged_file = f"responses/{dataset}/{dataset}_train_{model}_responses_merged.json"
    # Load merged summaries if exists
    if os.path.exists(merged_file):
        with open(merged_file, "r") as f:
            summaries = json.load(f)
    else:
        summaries = {}
    # Load articles and keys
    articles, keys = load_sources(dataset, extras=extras)
    all_keys = list(keys)
    # Determine missing keys
    missing_keys = [k for k in all_keys[:N] if k not in summaries]
    if not overwrite and len(summaries) >= N:
        print(f"[INFO] Already have {len(summaries)} responses for {model} in {dataset} (N={N}). Skipping.")
        return
    if overwrite:
        print(f"[INFO] Overwrite enabled. Will regenerate all {N} responses for {model} in {dataset}.")
        missing_keys = all_keys[:N]
        summaries = {}
    print(f"[INFO] {len(summaries)} existing responses found for {dataset}/{model} (target N={N}).")
    print(f"[INFO] {len(missing_keys)} missing responses to generate for {dataset}/{model}.")
    for key in tqdm(missing_keys, desc=f"Generating missing {dataset.upper()} responses for {model}"):
        summaries[key] = get_summary(articles[key], dataset, model)
        print(f"    [INFO] Generated response for key: {key}")
        print(summaries[key])
        with open(merged_file, "w") as f:
            json.dump(summaries, f, indent=2)
    print(f"[INFO] All responses for {dataset}/{model} saved to {merged_file}. Total: {len(summaries)}.")

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str)
    parser.add_argument("-N", type=int, default=1000)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--extras", action="store_true", default=False)
    args = parser.parse_args()

    TARGET = args.target.split(",")
    N = args.N or 1000
    main_models = TARGET
    datasets = args.datasets.split(",")

    print("Starting...")
    for model in main_models:
        for dataset in datasets:
            print(dataset)
            process_dataset(dataset, model, N, overwrite=args.overwrite, extras=args.extras)
        print(f"[INFO] {model} done!")
    print("Done!")

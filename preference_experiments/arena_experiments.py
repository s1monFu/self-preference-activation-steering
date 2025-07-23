import os
import glob
import pandas as pd
import argparse
import json
from math import exp
from tqdm import tqdm
from models import get_model_choice
from utils_config import (
    load_config_from_cli_and_file,
    generate_experiment_id,
    get_output_folder,
    save_config_and_metadata,
)
from utils_logging import get_logger

def normalize_model_name(name):
    # Remove hyphen after llama, add -fp8 for 70b and 405b
    if name.startswith('llama-'):
        name = name.replace('llama-', 'llama')
        if '70b' in name and not name.endswith('-fp8'):
            name += '-fp8'
        if '405b' in name and not name.endswith('-fp8'):
            name += '-fp8'
    return name

def process_csv(csv_path, model_name, output_folder, compare_type, overwrite, logger):
    df = pd.read_csv(csv_path)
    results = []
    glitches = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"[Arena] {model_name}"):
        question_id = row['id']
        self_model = normalize_model_name(row['self'])
        other_model = normalize_model_name(row['other'])
        prompt = row['prompt']
        self_response = row['self_response']
        other_response = row['other_response']
        won = row['won']
        # Forward: self vs other, Backward: other vs self
        try:
            forward_result = get_model_choice(
                self_response, other_response, prompt, compare_type, self_model, return_logprobs=True,
            )
            backward_result = get_model_choice(
                other_response, self_response, prompt, compare_type, self_model, return_logprobs=True,
            )
            forward_choice = forward_result[0].token
            backward_choice = backward_result[0].token
            forward_logprobs = forward_result[0].top_logprobs
            backward_logprobs = backward_result[0].top_logprobs
        except Exception as e:
            logger.warning(f"Error processing {question_id}: {e}")
            glitches += 1
            continue
        # Only keep if forward and backward agree
        if forward_choice != backward_choice:
            pick = 'self' if forward_choice == '1' else 'other'
            # Average exp(logprob) for first and second pick
            if pick == 'self':
                self_pick = [exp(forward_logprobs[0].logprob), exp(backward_logprobs[1].logprob)]
                other_pick = [exp(forward_logprobs[1].logprob), exp(backward_logprobs[0].logprob)]
            else:
                other_pick = [exp(forward_logprobs[1].logprob), exp(backward_logprobs[0].logprob)]
                self_pick = [exp(forward_logprobs[0].logprob), exp(backward_logprobs[1].logprob)]
            result = {
                'question_id': question_id,
                'reference_model': other_model,
                'pick': pick,
                'self_pick': sum(self_pick) / 2,
                'other_pick': sum(other_pick) / 2,
                'won': won
            }
            results.append(result)
    # Save results
    out_path = os.path.join(output_folder, f"{model_name}_arena_preference_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved arena results to {out_path}")

def load_arena_config(cli_args):
    """Custom config loader for arena preference tests."""
    # Only use CLI args and set dataset/models as needed
    config = dict(cli_args)
    config['dataset'] = 'arena'
    # Models will be inferred later from CSVs
    return config

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run arena-style preference experiments using model-based evaluation.\n\n"
            "This script compares model outputs in a head-to-head format using CSVs from arena_data/chat_arena/.\n"
            "For each model, it evaluates preferences between its own and other models' responses to prompts.\n"
            "Results are saved as JSON files in a timestamped output directory under arena_experiments/.\n\n"
            "Typical usage:\n"
            "  python arena_experiments.py --compare_type comparison_preference\n\n"
            "Arguments:\n"
            "  --compare_type   Type of comparison to use (default: comparison_preference)\n"
            "  --overwrite      Overwrite existing results\n"
            "  --log_level      Set logging level (e.g., INFO, DEBUG)\n"
            "  --config         (Unused, for compatibility)\n"
            "\n"
            "CSV files should be placed in arena_data/chat_arena/ and named as <model>_preference_data.csv.\n"
            "Each CSV should contain columns: id, self, other, prompt, self_response, other_response, won.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--compare_type", type=str, default="comparison_preference",
                        help="Type of comparison to use for model preference (default: comparison_preference)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results in the output directory")
    parser.add_argument("--log_level", type=str, default=None,
                        help="Set logging level (e.g., INFO, DEBUG)")
    parser.add_argument("--config", type=str, default=None,
                        help="(Unused, for compatibility with other scripts)")
    
    args = parser.parse_args()
    cli_args = vars(args)
    config = load_arena_config(cli_args)
    
    compare_type = config.get("compare_type", "comparison_preference")
    overwrite = config.get("overwrite", False)
    log_level = config.get("log_level", None)
    
    # Find all CSVs in chat_arena
    arena_dir = os.path.join(os.path.dirname(__file__), "arena_data", "chat_arena")
    csvs = glob.glob(os.path.join(arena_dir, "*.csv"))
    
    models = [normalize_model_name(os.path.basename(f).split("_preference_data")[0]) for f in csvs]
    config['models'] = models
    
    experiment_id = generate_experiment_id(dataset="arena", N=None, models=models)
    output_folder = get_output_folder("arena", experiment_id, base_dir="arena_experiments")
    
    os.makedirs(output_folder, exist_ok=True)
    save_config_and_metadata(config, output_folder)
    
    logger = get_logger(output_folder, log_level=log_level or "INFO")
    logger.info(f"Arena experiment started: {experiment_id}")
    logger.info(f"Models: {models}")
    
    for csv_path, model_name in zip(csvs, models):
        process_csv(csv_path, model_name, output_folder, compare_type, overwrite, logger)
    
    logger.info("Arena experiment complete.")

if __name__ == "__main__":
    main() 
import sys
from tqdm import tqdm
from data import load_data, save_to_json
from models import (
    get_model_choice,
    get_gpt_compare,
    code_datasets,
)
from math import exp
import json
import pandas as pd
import argparse
import os
from utils_config import (
    load_config_from_cli_and_file,
    generate_experiment_id,
    get_output_folder,
    save_config_and_metadata,
)
from utils_logging import get_logger
import glob
from plot_heatmap import make_heatmap_matrix


def simplify_compares(results, model_name_being_evaluated=None):
    """
    Processes comparison results to calculate mean confidence scores,
    detection accuracy, and self-preference rate for the model_name_being_evaluated
    when compared against various other models.

    Args:
        results (list): A list of dictionaries, where each dict is a comparison result.
                        Each result should contain:
                        - 'model': The name of the 'other' model in the comparison.
                        - 'detection_score': Confidence score for the detection task.
                        - 'self_preference': Confidence score for the preference task.
                        - 'forward_detection': The choice ('1' or '2') in the forward detection.
                                               '1' indicates source_summary (model_name_being_evaluated's) was chosen.
                        - 'backward_detection': The choice ('1' or '2') in the backward detection.
                                                '2' indicates source_summary (model_name_being_evaluated's) was chosen.
                        - 'forward_comparison': The choice ('1' or '2') in the forward comparison,
                                                adjusted for "worse" questions. '1' indicates source_summary was preferred.
        model_name_being_evaluated (str, optional): The name of the model whose results are being processed.
                                                   Used for potential future enhancements or logging.

    Returns:
        tuple: Contains four pandas Series:
            - mean_detect_confidence: Mean detection confidence against each 'other' model.
            - mean_prefer_confidence: Mean self-preference confidence against each 'other' model.
            - detection_accuracy: Detection accuracy against each 'other' model, considering both
                                  forward and backward detection passes as opportunities.
            - self_preference_rate: Self-preference rate against each 'other' model.
    """
    
    detect_confidences = {}
    prefer_confidences = {}
    correct_detection_counts = {} # Sum of correct detections in both forward and backward passes
    self_preference_counts = {}
    total_individual_comparisons = {} # Counts unique (source_summary, other_summary) pairs

    for result in results:
        other_model = result['model'] 

        if other_model not in total_individual_comparisons:
            detect_confidences[other_model] = []
            prefer_confidences[other_model] = []
            correct_detection_counts[other_model] = 0
            self_preference_counts[other_model] = 0
            total_individual_comparisons[other_model] = 0
        if result.get("forward_detection") == result.get("backward_detection"):
            continue
        total_individual_comparisons[other_model] += 1
        detect_confidences[other_model].append(result['detection_score'])
        prefer_confidences[other_model].append(result['self_preference'])
        
        
        # Check for correct detection in the forward pass
        if result.get('forward_detection') == '1':
            correct_detection_counts[other_model] += 1
        
        # Check for correct detection in the backward pass
        # (source_summary is choice '2' in the backward pass configuration)
        if result.get('backward_detection') == '2':
            correct_detection_counts[other_model] += 1
            
        # Check for self-preference
        if result.get('forward_comparison') == '1':
            self_preference_counts[other_model] += 1
        
        if result.get('backward_comparison') == '2':
            self_preference_counts[other_model] += 1
    
    mean_detect_confidence_data = {model: pd.Series(scores).mean() for model, scores in detect_confidences.items()}
    mean_prefer_confidence_data = {model: pd.Series(scores).mean() for model, scores in prefer_confidences.items()}

    mean_detect_confidence = pd.Series(mean_detect_confidence_data, name="mean_detection_confidence")
    mean_prefer_confidence = pd.Series(mean_prefer_confidence_data, name="mean_self_preference_confidence")

    detection_accuracy_data = {}
    for model, count in correct_detection_counts.items():
        # Each individual comparison offers two detection opportunities (forward and backward)
        num_detection_opportunities = total_individual_comparisons.get(model, 0) * 2
        detection_accuracy_data[model] = count / num_detection_opportunities if num_detection_opportunities > 0 else 0.0
    detection_accuracy = pd.Series(detection_accuracy_data, name="detection_accuracy")
    
    self_preference_rate_data = {}
    for model, count in self_preference_counts.items():
        # Self-preference is one outcome per individual comparison
        total_prefs = total_individual_comparisons.get(model, 0) * 2
        self_preference_rate_data[model] = count / total_prefs if total_prefs > 0 else 0.0
    self_preference_rate = pd.Series(self_preference_rate_data, name="self_preference_rate")
    return mean_detect_confidence, mean_prefer_confidence, detection_accuracy, self_preference_rate


def aggregate_existing_results(result_json_path):
    """
    Load existing results from a JSON file if it exists, else return an empty list.
    This helps avoid recomputation if results are already present.

    Args:
        result_json_path (str): Path to the JSON file containing results.

    Returns:
        list: List of result dictionaries, or empty list if file does not exist or is invalid.
    """
    if os.path.exists(result_json_path):
        with open(result_json_path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

def get_existing_keys(results, key_field="key"):
    """
    Get a set of keys already processed in the results.
    Used to skip already-completed experiments.

    Args:
        results (list): List of result dictionaries.
        key_field (str): The field name to use as the key (default: "key").

    Returns:
        set: Set of keys found in the results.
    """
    return set(r[key_field] for r in results)

def find_existing_result(dataset, model, reference, key, search_dirs):
    """
    Search for an existing result for (model, reference, key) in the given directories.
    Enables result reuse across experiment runs and folders.

    Args:
        dataset (str): Name of the dataset.
        model (str): Name of the model being evaluated.
        reference (str): Name of the reference model.
        key (str): Unique identifier for the example.
        search_dirs (list): List of directories to search for results.

    Returns:
        dict or None: The result dictionary if found, else None.
    """
    for search_dir in search_dirs:
        if "individual_setting" in search_dir:
            pattern = os.path.join(search_dir, f"{model}_comparison_results*.json")
        else:
            pattern = os.path.join(search_dir, model, f"{model}_comparison_results*.json")
        for file in glob.glob(pattern):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                for entry in data:
                    if entry["model"] == reference and entry["key"] == key:
                        entry['source'] = file
                        return entry
            except Exception:
                continue
    return None

if __name__ == "__main__":
    # 1. Parse CLI arguments
    # Set up argument parser for experiment configuration
    # Allows flexible control from the command line or scripts
    parser = argparse.ArgumentParser(description="Run model experiments with reproducible config and logging.")
    parser.add_argument("--dataset", type=str, required=False, default=None, help="Name of the dataset to use (e.g., 'cnn', 'xsum', 'apps', etc.)")
    parser.add_argument("--models", type=str, required=False, default=None, help="Comma-separated list of model names to evaluate (e.g., 'llama3,deepseek,hermes3')")  # comma-separated
    parser.add_argument("--references", type=str, required=False, default=None, help="Comma-separated list of reference models to compare against (default: same as --models)")
    parser.add_argument("--N", type=int, default=None, help="Number of examples to use from the dataset (default: all available)")
    parser.add_argument("--compare_type", type=str, default=None, help="Type of comparison to run (e.g., 'comparison', 'comparison_preference')")
    parser.add_argument("--detection_type", type=str, default=None, help="Type of detection task to run (e.g., 'detection', 'detection_code')")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if present")
    parser.add_argument("--log_level", type=str, default=None, help="Logging level (e.g., 'INFO', 'DEBUG')")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for model calls (default: 120)")
    parser.add_argument("--max_retries", type=int, default=None, help="Maximum number of retries for model calls (default: 10)")
    parser.add_argument("--use_existing_results", action="store_true", default=None, help="Reuse existing results if available to avoid recomputation (default False)")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file with experiment settings. This will provide the default arguments, which other CLI options override.")
    args = parser.parse_args()

    # Convert CLI args to dict, handling comma-separated models and references
    # This enables passing multiple models/references as comma-separated strings
    cli_args = vars(args)
    if cli_args["models"]:
        cli_args["models"] = [m.strip() for m in cli_args["models"].split(",")]
    if cli_args["references"]:
        cli_args["references"] = [m.strip() for m in cli_args["references"].split(",")]

    # 2. Load config and metadata (defaults -> config file -> CLI)
    # Loads experiment configuration, prioritizing CLI > config file > defaults
    config = load_config_from_cli_and_file(cli_args, config_file_path=args.config)

    # Assert required fields
    # Ensures that dataset and models are specified before proceeding
    if not config.get("dataset") or not config.get("models"):
        raise ValueError("'dataset' and 'models' must be specified in config or CLI.")
    if not config.get("references"):
        config["references"] = config["models"]

    # Generate experiment ID and output folder for reproducibility and organization
    experiment_id = generate_experiment_id(
        dataset=config["dataset"], N=config["N"], models=config["models"]
    )
    output_folder = get_output_folder(config["dataset"], experiment_id)
    save_config_and_metadata(config, output_folder)

    # 3. Set up logging
    # All experiment logs are saved to the output folder for traceability
    logger = get_logger(output_folder, log_level=config["log_level"])
    logger.info(f"Experiment started: {experiment_id}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # 4. Main experiment logic
    # Extract config values for use in the experiment
    models = config["models"]
    references = config["references"]
    N = config["N"]
    dataset = config["dataset"]
    compare_type = config["compare_type"]
    detection_type = config["detection_type"]
    if dataset in code_datasets:
        detection_type += "_code"
        compare_type += "_code"
    overwrite = config["overwrite"]
    use_existing_results = config["use_existing_results"]

    # Prepare search directories for result reuse
    # This allows the experiment to find and reuse results from previous runs
    search_dirs = [output_folder]
    # Add all previous experiment folders for this dataset
    exp_dataset_dir = os.path.join("experiments", dataset)
    if os.path.exists(exp_dataset_dir):
        for d in os.listdir(exp_dataset_dir):
            full_path = os.path.join(exp_dataset_dir, d)
            if os.path.isdir(full_path) and full_path != output_folder:
                search_dirs.append(full_path)
    # Add legacy score_results folder
    legacy_dir = os.path.join("individual_setting", "score_results", dataset)
    if os.path.exists(legacy_dir):
        search_dirs.append(legacy_dir)

    # Load data once for all models
    # responses: dict of model -> key -> response
    # articles: dict of key -> article text
    # keys: list of unique identifiers for each example
    logger.info(f"Loading data for dataset: {dataset}, N={N}, models={models}")
    responses, articles, keys = load_data(dataset, sources=references, target_model=models[0], num_samples=N, logger=logger)
    logger.info(f"Loaded {len(keys)} keys for dataset {dataset}")

    for model in models:
        model_folder = os.path.join(output_folder, model)
        os.makedirs(model_folder, exist_ok=True)
        logger.info(f"Processing model: {model}")
        # Comparison experiment: check if results already exist, else run experiment
        comparison_json_path = os.path.join(model_folder, f"{model}_comparison_results.json")
        if not overwrite and os.path.exists(comparison_json_path):
            logger.info(f"Skipping {comparison_json_path} (already exists)")
            with open(comparison_json_path, "r") as f:
                results = json.load(f)
        else:
            logger.info(f"Running comparison experiment for {model}")
            results = []
            glitches = 0
            # Iterate over all keys/examples
            for key in tqdm(keys, desc=f"[Comparison] {model}"):
                article = articles[key]
                source_summary = responses[model][key]
                # Compare against all other reference models
                for other in [s for s in references if s != model]:
                    # Try to reuse result if available
                    if use_existing_results:
                        existing = find_existing_result(dataset, model, other, key, search_dirs)
                        if existing:
                            logger.info(f"Reusing result for ({model}, {other}, {key})")
                            results.append(existing)
                            continue
                    result = {"key": key, "model": other}
                    other_summary = responses[other][key]
                    
                    # Detection: model tries to distinguish its own output from another's
                    forward_result = get_model_choice(
                        source_summary, other_summary, article, detection_type, model, return_logprobs=True,
                    )
                    backward_result = get_model_choice(
                        other_summary, source_summary, article, detection_type, model, return_logprobs=True,
                    )
                    forward_choice = forward_result[0].token
                    backward_choice = backward_result[0].token
                    forward_result = forward_result[0].top_logprobs
                    backward_result = backward_result[0].top_logprobs
                    result["forward_detection"] = forward_choice
                    result["forward_detection_probability"] = exp(forward_result[0].logprob)
                    result["backward_detection"] = backward_choice
                    result["backward_detection_probability"] = exp(backward_result[0].logprob)
                    result["source"] = comparison_json_path
                    # Score: aggregate detection probabilities depending on choices
                    match (forward_choice, backward_choice):
                        case ("1", "2"):
                            result["detection_score"] = 0.5 * (
                                exp(forward_result[0].logprob) + exp(backward_result[0].logprob)
                            )
                        case ("2", "1"):
                            result["detection_score"] = 0.5 * (
                                exp(forward_result[1].logprob) + exp(backward_result[1].logprob)
                            )
                        case ("1", "1"):
                            result["detection_score"] = 0.5 * (
                                exp(forward_result[0].logprob) + exp(backward_result[1].logprob)
                            )
                        case ("2", "2"):
                            result["detection_score"] = 0.5 * (
                                exp(forward_result[1].logprob) + exp(backward_result[0].logprob)
                            )
                    # Comparison: model chooses which summary is better
                    forward_result = get_model_choice(
                        source_summary, other_summary, article, compare_type, model, return_logprobs=True,
                    )
                    backward_result = get_model_choice(
                        other_summary, source_summary, article, compare_type, model, return_logprobs=True,
                    )
                    
                    # Loading and storing data object ('result')
                    forward_choice = forward_result[0].token
                    backward_choice = backward_result[0].token
                    forward_result = forward_result[0].top_logprobs
                    backward_result = backward_result[0].top_logprobs
                    result["forward_comparison"] = forward_choice
                    result["forward_comparison_probability"] = exp(forward_result[0].logprob)
                    result["backward_comparison"] = backward_choice
                    result["backward_comparison_probability"] = exp(backward_result[0].logprob)
                    # Aggregate preference scores
                    match (forward_choice, backward_choice):
                        case ("1", "2"):
                            result["self_preference"] = 0.5 * (
                                exp(forward_result[0].logprob) + exp(backward_result[0].logprob)
                            )
                        case ("2", "1"):
                            result["self_preference"] = 0.5 * (
                                exp(forward_result[1].logprob) + exp(backward_result[1].logprob)
                            )
                        case ("1", "1"):
                            result["self_preference"] = 0.5 * (
                                exp(forward_result[0].logprob) + exp(backward_result[1].logprob)
                            )
                        case ("2", "2"):
                            result["self_preference"] = 0.5 * (
                                exp(forward_result[1].logprob) + exp(backward_result[0].logprob)
                            )
                        case _:
                            glitches += 1
                            continue
                    logger.info(f"Computed new result for ({model}, {other}, {key})")
                    results.append(result)
            # Save all results for this model
            save_to_json(results, comparison_json_path)
            logger.info(f"Saved comparison results to {comparison_json_path}")
        # Simplify and save metrics for this model
        mean_dc, mean_pc, detect_acc, prefer_rate = simplify_compares(
            results, model_name_being_evaluated=model
        )
        mean_dc.to_csv(os.path.join(model_folder, f"{model}_comparison_results_mean_detect_conf_simple.csv"), header=True)
        mean_pc.to_csv(os.path.join(model_folder, f"{model}_comparison_results_mean_prefer_conf_simple.csv"), header=True)
        detect_acc.to_csv(os.path.join(model_folder, f"{model}_comparison_results_detect_accuracy_simple.csv"), header=True)
        prefer_rate.to_csv(os.path.join(model_folder, f"{model}_comparison_results_self_prefer_rate_simple.csv"), header=True)
        logger.info(f"Saved simplified metrics for {model}")
        # Sanity check: ensure expected number of results were produced
        expected_results = len(keys) * (len(references) - 1) - glitches
        assert len(results) == expected_results, f"Expected {expected_results} results for model {model}, got {len(results)}"
        logger.info(f"Sanity check passed: {len(results)} results for {model}")

    # Generate heatmap for self_preference_rate
    # Visualizes how often each model prefers its own output
    logger.info("Generating self-preference rate heatmap for this experiment...")
    make_heatmap_matrix(
        exp_dir=output_folder,
        metric='self_preference_rate'
    )

    # Generate heatmap for detection_accuracy
    # Visualizes how well each model can detect its own output
    logger.info("Generating detection accuracy heatmap for this experiment...")
    make_heatmap_matrix(
        exp_dir=output_folder,
        metric='detection_accuracy'
    )

    logger.info("Experiment complete.")

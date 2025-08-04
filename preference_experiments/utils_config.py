import os
import json
import datetime
import getpass
import subprocess

DEFAULT_CONFIG = {
    "models": None,  # required
    "N": 1000,
    "compare_type": "comparison",
    "detection_type": "detection",
    "overwrite": False,
    "log_level": "INFO",
    "timeout": 120,
    "max_retries": 10,
    "use_existing_results": True,
}

def load_config_from_cli_and_file(cli_args, config_file_path=None):
    """
    Load config from a file if provided, otherwise from CLI args, applying defaults and validating required fields.
    Args:
        cli_args (dict): Dictionary of CLI arguments.
        config_file_path (str): Path to config file (optional).
    Returns:
        config (dict): Final config dictionary.
    """
    config = DEFAULT_CONFIG.copy()
    if config_file_path:
        with open(config_file_path, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    # CLI args override config file (including None for flags)
    for k, v in cli_args.items():
        if v is not None:
            config[k] = v
    # Assert required fields
    if not config.get("dataset") or not config.get("models"):
        raise ValueError("'dataset' and 'models' must be specified in config or CLI.")
    # Handle references default
    if not config.get("references"):
        config["references"] = config["models"]
    return config

def generate_experiment_id(dataset, N, models, timestamp=None):
    """
    Generate a unique, interpretable experiment ID.
    Args:
        dataset (str): Dataset name.
        N (int): Number of samples.
        models (list): List of model names.
        timestamp (str): Optional timestamp (ISO format or None for now).
    Returns:
        experiment_id (str)
    """
    if timestamp is None:
        # Use up to the minute for readability and to avoid issues with seconds
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    num_models = len(models)
    return f"{dataset}{'_n' + str(N) if N is not None else ''}_m{num_models}_{timestamp}"

def get_output_folder(dataset, experiment_id, base_dir="experiments"):
    """
    Get the output folder path for the experiment.
    Args:
        dataset (str): Dataset name.
        experiment_id (str): Experiment ID.
        base_dir (str): Base directory for experiments.
    Returns:
        output_folder (str)
    """
    return os.path.join(base_dir, dataset, experiment_id)

def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return commit
    except Exception:
        return None

def save_config_and_metadata(config, output_folder, result_filepaths=None):
    """
    Save config.json and metadata.json to the output folder.
    Args:
        config (dict): Experiment config.
        output_folder (str): Path to output folder.
        result_filepaths (dict): Dict of result files (optional, can be filled in later).
    Returns:
        metadata_path (str)
    """
    os.makedirs(output_folder, exist_ok=True)
    config_path = os.path.join(output_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    metadata = {
        "experiment_id": os.path.basename(output_folder),
        "dataset": config.get("dataset"),
        "models": config.get("models"),
        "references": config.get("references", config.get("models:")),
        "N": config.get("N"),
        "compare_type": config.get("compare_type"),
        "detection_type": config.get("detection_type"),
        "overwrite": config.get("overwrite"),
        "log_level": config.get("log_level"),
        "timeout": config.get("timeout"),
        "max_retries": config.get("max_retries"),
        "use_existing_results": config.get("use_existing_results"),
        "result_filepaths": result_filepaths or {},
        "timestamp": datetime.datetime.now().isoformat(),
    }
    metadata_path = os.path.join(output_folder, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path
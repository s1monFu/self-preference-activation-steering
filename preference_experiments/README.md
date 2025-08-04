
# Preference Experiments
_written by Dani Roytburg, last updated July 22nd, 2025_

![Graphic depicting a language model looking at itself lovingly in the mirror.](../assets/martian-graphic.png)

<p align="right"><i>graphic courtesy of Antia@Martian Research</i></p>

This folder contains scripts and utilities for running and analyzing **self-preference experiments** on large language models hosted through the Lambda Inference API, including baseline preference testing, arena-style experiments, and result aggregation.

## Pre-requisites
A Lambda Inference API Key is the only thing necessary for setup. You may also use any other key/URL provider that can work with the OpenAI API. 

- Set environment variables:
  - LAMBDA_API_KEY (can be any OpenAI API-compatible API Key)
  - LAMBDA_API_URL (optional, if not filled set with the Lambda default)

Installs are kept in `requirements.txt`, and can be installed as usual with:

``pip install -r requirements.txt``

## Usage

### 1. Baseline Preference Testing

To run **baseline preference experiments** (e.g., does model A prefer model A or model B?) on a set of prompts that do not have "objective" answers (no gold- or silver- labels):

```
python3 experiments.py --dataset cnn --models "llama-3.1-8b-instruct,gpt-3.5,deepseek-v3-0324" --references "llama-3.1-8b-instruct,llama3.2-3b-instruct,hermes3-405b,deepseek-v3-0324" --N 200 --use_existing_results 
```

We also include some basic **config files** to reference. If you want to reproduce our results on the CNN summarization task, override defaults with:

```
python3 experiments.py --config configs/cnn_5models_300entries_lambda_config.json
```

### 2. Ground-Truth Testing with LMArena data

To run **preference testing with ground-truth** labels, primarily using the Arena experiments, you can simply run 

```
python3 arena_experiments.py
```

## Folder Overview

- **experiments.py**  
  Main entry point for running baseline preference experiments. Generates results for model preference between summaries or prompts.

- **arena_experiments.py**  
  Runs "arena" style experiments, where models are evaluated in a head-to-head format to compare preferences across multiple prompts or settings.

- **aggregate_arena_results.py**  
  Aggregates and summarizes results from multiple arena experiment runs, producing overall statistics and summary tables.

- **analysis.ipynb**  
  Jupyter notebook for in-depth analysis and visualization of experiment results.

- **comparison_results_to_steering_data.ipynb**  
  Notebook for comparing preference experiment results to steering vector data.

- **generate_summaries.py**  
  Script for generating model summaries for use in preference experiments.

- **merge_summaries.py**  
  Utility for merging multiple summary files into a single dataset.

- **models.py**  
  Contains model loading and inference utilities.

- **plot_heatmap.py**  
  Script for visualizing results as heatmaps.

- **prompts.py**  
  Contains prompt templates and prompt generation logic.

- **utils_config.py**  
  Configuration utilities for experiments.

- **utils_logging.py**  
  Logging utilities for experiment scripts.

- **arena_data/**  
  Contains data and results from arena experiments.

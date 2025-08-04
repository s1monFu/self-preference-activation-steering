import os
import re
import glob
import json
from datasets import load_dataset
from tqdm import tqdm
from models import GPT_MODEL_ID, code_datasets
from generate_summaries import process_dataset


def save_to_json(dictionary, file_name, force_overwrite=True):
    """Save a dictionary to a JSON file, creating directories if needed."""
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    if not force_overwrite and os.path.exists(file_name):
        return

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    """Load a dictionary from a JSON file."""
    with open(file_name, "r") as f:
        return json.load(f)


def load_data(dataset, sources, target_model, num_samples, load_responses=True, extras=False, logger=None):
    """
    Load responses and source data for a given dataset and set of sources.
    Returns (responses, sources, keys).
    """
    all_responses = {}
        
    print(num_samples)
    data_type = "sources" if dataset not in code_datasets else "code" #in case we want to add other types of sources in the future
    if load_responses:
        for source_model in sources:
            print(f"[DEBUG] Loading {dataset} data for model: {source_model}")
            merged_file = f"responses/{dataset}/{dataset}_train_{source_model}_responses_merged.json"
            print(f"[DEBUG] Checking merged file: {merged_file}")
            if os.path.exists(merged_file):
                print(f"[DEBUG] Merged file exists, loading...")
                responses = load_from_json(merged_file)
                print(f"[DEBUG] Loaded {len(responses)} samples from merged file.")
                if len(responses) < num_samples:
                    print(f"[DEBUG] Not enough samples: {len(responses)} < {num_samples}. Generating {num_samples - len(responses)} more now.")
                    process_dataset(dataset, source_model, num_samples)
                print(f"[DEBUG] Using merged file for {source_model}")
                all_responses[source_model] = responses
            else:
                # Fallback: find the best available non-merged file with at least num_samples
                pattern = f"responses/{dataset}/{dataset}_train_{source_model}_responses*{'_extra' if extras else ''}.json"
                files = glob.glob(pattern)
                best_file = None
                for file in files:
                    with open(file, "r") as f:
                        data = json.load(f)
                    if len(data) >= num_samples:
                        best_file = file
                        break
                if best_file is None:
                    if logger is not None:
                        logger.warning(f"No suitable {data_type} file found for {source_model} with at least {num_samples} samples. Generating now.")
                    else:
                        print(f"No suitable {data_type} file found for {source_model} with at least {num_samples} samples. Generating now.")
                    process_dataset(dataset, source_model, num_samples)
                    # After generation, check for the merged file first, then fallback files
                    merged_file = f"responses/{dataset}/{dataset}_train_{source_model}_responses_merged.json"
                    if os.path.exists(merged_file):
                        best_file = merged_file
                    else:
                        # Check for newly generated files
                        pattern = f"responses/{dataset}/{dataset}_train_{source_model}_responses*{'_extra' if extras else ''}.json"
                        files = glob.glob(pattern)
                        for file in files:
                            with open(file, "r") as f:
                                data = json.load(f)
                            if len(data) >= num_samples:
                                best_file = file
                                break
                        if best_file is None:
                            raise FileNotFoundError(f"Failed to generate or find suitable {data_type} file for {source_model} with at least {num_samples} samples.")
                all_responses[source_model] = load_from_json(best_file)
    articles = load_from_json(f"{data_type}/{dataset}_train_{data_type}{'_extra' if extras else ''}.json")
    if target_model in sources:
        print(all_responses.keys())
        all_keys = list(all_responses[target_model].keys())
        keys = all_keys[:num_samples]
    elif load_responses:
        raise Exception("Model not found!", target_model)
    else:
        keys = list(articles.keys())
    return all_responses, articles, keys


def load_cnn_dailymail_data():
    """
    Load the CNN/DailyMail dataset splits.
    Returns (train_data, test_data, validation_data).
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_data = dataset["train"]
    test_data = dataset["test"]
    validation_data = dataset["validation"]
    return train_data, test_data, validation_data


def load_xsum_data():
    """
    Load the XSum dataset splits.
    Returns (train_data, test_data, validation_data).
    """
    dataset = load_dataset("EdinburghNLP/xsum")
    train_data = dataset["train"]
    test_data = dataset["test"]
    validation_data = dataset["validation"]
    return train_data, test_data, validation_data



def load_arena_data():
    data = {}
    arena = load_dataset("lmarena-ai/arena-human-preference-100k",split="train")
    for model in ["llama-3.1-8b-instruct", "llama-3.1-70b-instruct", "llama-3.1-405b-instruct"]:
        data[model] = arena.filter(lambda a: a['model_a'] == model or a['model_b'] == model)
    for model in ["llama-3.1-8b-instruct", "llama-3.1-70b-instruct", "llama-3.1-405b-instruct"]:
        model_data = []
        for entry in data[model]:
            add = {}
            assert entry['conversation_a'][0]['content'] == entry['conversation_b'][0]['content']
            if not len(entry['conversation_a']) == len(entry['conversation_b']) == 2:
                ia += 1
                for i in range(0, len(entry['conversation_a']), 2):
                    a, b = entry['conversation_a'][i], entry['conversation_b'][i]
                    assert a['content'] == b['content']
                continue

            own = 'a' if entry['model_a'] == model else 'b'
            other = 'b' if own == 'a' else 'a'

            add['id'] = entry['question_id']
            add['self'] = entry[f"model_{own}"]
            add['other'] = entry[f'model_{other}']
            add['prompt'] = entry[f'conversation_{own}'][0]['content']
            add['self_response'] = entry[f'conversation_{own}'][-1]['content']
            add['other_response'] = entry[f'conversation_{other}'][-1]['content']
            add['won'] = 1 if entry['winner'] == f"model_{own}" else 0
            add['language'] = entry['language']

            model_data.append(add)
        return model_data

def load_medmcqa_data():
    medmcqa = load_dataset("openlifescienceai/medmcqa")
    medmcqa_json = {}; medmcqa_answers = {}
    ref = ['a','b','c','d']
    for example in tqdm(medmcqa['train']):
        correct_answer = 'op' + ref[example['cop']]
        medmcqa_json[example['id']] = \
        f"""{example['question']}
            Correct Answer: {example[correct_answer]}

            Can you explain why this is the case?
        """
        print(medmcqa_json[example['id']])
        medmcqa_answers[example['id']] = example['exp']
    with open("../sources/medmcqa_train_sources.json","w") as f:
        json.dump(medmcqa_json, f)
    with open("../responses/medmcqa/medmcqa_train_human_responses_merged.json","w") as f:
        json.dump(medmcqa_answers, f)

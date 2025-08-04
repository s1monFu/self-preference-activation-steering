from transformers import *
from datasets import load_dataset

import pandas as pd
import json
from tqdm import tqdm


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

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from pprint import pprint
import json
import os
from time import sleep

import openai

from prompts import (
    COMPARISON_PROMPT_TEMPLATE_CODE,
    COMPARISON_PROMPT_TEMPLATE_PREFERENCE,
    COMPARISON_SYSTEM_PROMPT_CODE,
    COMPARISON_SYSTEM_PROMPT_PREFERENCE,
    DATASET_SYSTEM_PROMPTS,
    COMPARISON_PROMPT_TEMPLATE,
    COMPARISON_SYSTEM_PROMPT,
    DETECTION_PROMPT_TEMPLATE,
    DETECTION_PROMPT_TEMPLATE_CODE,
    DETECTION_PROMPT_TEMPLATE_VS_HUMAN,
    DETECTION_PROMPT_TEMPLATE_VS_MODEL,
    DETECTION_SYSTEM_PROMPT,
    COMPARISON_PROMPT_TEMPLATE_WITH_SOURCES,
    COMPARISON_PROMPT_TEMPLATE_WITH_WORSE,
    DETECTION_SYSTEM_PROMPT_CODE,
    SCORING_SYSTEM_PROMPT,
    RECOGNITION_SYSTEM_PROMPT,
    RECOGNITION_PROMPT_TEMPLATE,
)

code_datasets = ["apps"]


GPT_MODEL_ID = {
    "gpt4": "gpt-4-1106-preview",
    "gpt35": "gpt-3.5-turbo-1106",
    "xsum_2_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8nc8TgDp",
    "xsum_10_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8nYmytb4",
    "xsum_500_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8kP7i66k",
    "xsum_always_1_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8nZloDpW",
    "xsum_random_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8nZloDpW",
    "xsum_readability_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8oLO7FOF",
    "xsum_length_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8ooNDQYs",
    "xsum_vowelcount_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8ooNNbtT",
    "cnn_2_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rX9zfcC",
    "cnn_10_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rXDPMYM",
    "cnn_500_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rYivqW8",
    "cnn_always_1_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rYwud4k",
    "cnn_random_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rYvYVKD",
    "cnn_readability_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rbOOAw9",
    "cnn_length_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rbPCDli",
    "cnn_vowelcount_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8raOJ2nT",
    "llama3.3-70b-instruct-fp8": "llama3.3-70b-instruct-fp8",
    "llama3.1-70b-instruct-fp8": "llama3.1-70b-instruct-fp8",
    "llama-4-scout-17b-16e-instruct": "llama-4-scout-17b-16e-instruct",
    "llama3.1-8b-instruct":"llama3.1-8b-instruct"
}

load_dotenv()
lambda_api_key = os.getenv("LAMBDA_API_KEY")
lambda_api_base = os.getenv("LAMBDA_API_URL") or "https://api.lambda.ai/v1"

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_BASE_URL")

martian_api_key = os.getenv("MARTIAN_API_KEY")
martian_api_base = os.getenv("MARTIAN_API_URL")

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_api_base = os.getenv("GEMINI_BASE_URL")

# Initialize the OpenAI client
openai_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    timeout=120,
    max_retries=10
)
if openai_client is None:
    exit()

def init_client(model):
    if "gpt" in model.lower() and False: #using martian client
        print("Using Martian as Provider")

        return OpenAI(
            api_key=martian_api_key,
            base_url= martian_api_base + "/openai/v2",
            timeout=120,
            max_retries=10
        )
    elif "gpt" in model.lower():
        print("Using OpenAI as Provider")
        return OpenAI(
            api_key=openai_api_key,
            base_url= openai_api_base,
            timeout=120,
            max_retries=10
        )
    elif "gemini" in model.lower():
        print("Using Google as Provider")
        return OpenAI(
            api_key=gemini_api_key,
            base_url=gemini_api_base,
            timeout=120,
            max_retries=10
        )
    else:
        print("Using Lambda API as Provider")
        return OpenAI(
            api_key=lambda_api_key,
            base_url=lambda_api_base,
            timeout=120,
            max_retries=10
        )
anthropic_client = anthropic.Anthropic()


def get_gpt_summary(article, dataset, model) -> str:
    history = [
        {"role": "system", "content": DATASET_SYSTEM_PROMPTS[dataset]},
        {
            "role": "user",
            "content": f"Article:\n{article}\n\nProvide only the summary with no other text.",
        },
    ]
    attempts = 0
    while attempts < 10:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=history,
                max_tokens=100,
                temperature=0
            )
            if not isinstance(response.choices[0].message.content, str):
                print("List format")
                return response.choices[0].message.content[0]
            elif isinstance(response.choices[0].message.content, str):
                print("String format")
                return response.choices[0].message.content
            else:
                print("Unexpected content format:", response.choices[0].message.content)
                raise ValueError("Unexpected content format of type " + str(type(response.choices[0].message.content)))
            return response.choices[0].message.content
        except openai.APITimeoutError:
            attempts += 1
            sleep(5)
            print(f"Timeout error after {attempts} attempts, retrying...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sleep(5)
            raise e
    return response.choices[0].message.content


def get_code(codebase, dataset, model, pipe=None):
    
    apk = openai_api_key if "gpt" in model.lower() else gemini_api_key if "gemini" in model.lower() else lambda_api_key
    global openai_client
    if openai_client.api_key != apk:
        print("Need new client")
        openai_client = init_client(model)
    if model == "claude":
        return get_claude_summary(
            codebase,
            dataset,
        )
    if model == "gpt4":
        return get_gpt_code(codebase, dataset, model="gpt-4-1106-preview")
    if model.endswith("gpt35"):
        return (
            get_gpt_code(
                codebase,
                dataset,
                model=GPT_MODEL_ID[model],
            ),
        )

    else:
        return (
            get_gpt_code(
                codebase,
                dataset,
                model=model,
            ),
        )

def get_gpt_code(code, dataset, model):
    history = [
        {"role": "system",
         "content": DATASET_SYSTEM_PROMPTS[dataset]},
        {"role": "user",
         "content": f"Codebase:\n{code}\n\nProvide the answer in Python with no other text."}
        ]
    response = openai_client.chat.completions.create(
            model=model,
            messages=history,
            max_tokens=300,
            temperature=0,

        )
    print(response.choices[0])
    return response.choices[0].message.content

def get_summary(article, dataset, model, pipe=None):
    global openai_client
    assert openai_client is not None
    apk = openai_api_key if "gpt" in model.lower() else gemini_api_key if "gemini" in model.lower() else lambda_api_key
    if openai_client.api_key != apk:
        print("Need new client")
        openai_client = init_client(model)
    if model == "claude":
        return get_claude_summary(
            article,
            dataset,
        )
    if model == "gpt4":
        return get_gpt_summary(article, dataset, model="gpt-4-1106-preview")
    if model.endswith("gpt35"):
        return get_gpt_summary(
                article,
                dataset,
                model=GPT_MODEL_ID[model],
            )
    # elif "llama" in model.lower():
    #     return (
    #         get_llama_summary(
    #             article,
    #             dataset,
    #             pipe,
    #         )
    #     )
    else:
        return get_gpt_summary(
                article,
                dataset,
                model=model,
            )


def get_claude_summary(article, dataset="xsum"):
    response_type = "highlights" if dataset in ["cnn", "dailymail"] else "summary"
    message = anthropic_client.beta.messages.create(
        model="claude-2.1",
        max_tokens=100,
        system=DATASET_SYSTEM_PROMPTS[dataset],
        messages=[
            {
                "role": "user",
                "content": f"Article:\n{article}\n\nProvide only the {response_type} with NO other text.",
            }
        ],
    )
    return message.content[0].text


def get_claude_choice(summary1, summary2, article, choice_type) -> str:
    match choice_type:
        case "comparison":
            prompt = COMPARISON_PROMPT_TEMPLATE.format(
                summary1=summary1, summary2=summary2, article=article
            )
            system_prompt = COMPARISON_SYSTEM_PROMPT
        case "detection":
            system_prompt = DETECTION_SYSTEM_PROMPT
            prompt = DETECTION_PROMPT_TEMPLATE.format(
                summary1=summary1, summary2=summary2, article=article
            )

    message = anthropic_client.beta.messages.create(
        model="claude-2.1",
        max_tokens=10,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text

#For explaining differences, not for measuring between them
def get_gpt_compare(
    summary1,
    summary2,
    article,
    model="gpt4-1106-preview",
) -> str:
    prompt = f"""Here are two different summmaries for an article:
                \n\n Article: {article}
                \n\n Summary One: {summary1}
                \n\n Summary Two: {summary2}.
                \n\n Which one is better, and why?
                """
    history = [
        {"role": "system", "content": "You are comparing two different outputs for a news summarization task."},
        {"role": "user", "content": prompt},
    ]
    attempts = 0
    while attempts < 10:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=history,
                max_tokens=100,
            )
            return response.choices[0].message.content
        except openai.APITimeoutError:
            attempts += 1
            sleep(5)
            print(f"Timeout error after {attempts} attempts, retrying...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sleep(5)
            return "1"
    print(f"Failed after {attempts} attempts.")
    return "1"


def get_gpt_choice(
    summary1,
    summary2,
    article,
    choice_type,
    model="gpt4-1106-preview",
    return_logprobs=False,
) -> str:
    match choice_type:
        case "comparison":
            prompt = COMPARISON_PROMPT_TEMPLATE.format(
                summary1=summary1, summary2=summary2, article=article
            )
            system_prompt = COMPARISON_SYSTEM_PROMPT
        case "comparison_preference":
            prompt = COMPARISON_PROMPT_TEMPLATE_PREFERENCE.format(
                summary1=summary1, summary2=summary2, article=article
            )
            system_prompt = COMPARISON_SYSTEM_PROMPT_PREFERENCE
        case "comparison_code":
            prompt = COMPARISON_PROMPT_TEMPLATE_CODE.format(
                snippet1=summary1, snippet2=summary2, request=article
            )
            system_prompt = COMPARISON_SYSTEM_PROMPT_CODE
        case "comparison_with_worse":
            prompt = COMPARISON_PROMPT_TEMPLATE_WITH_WORSE.format(
                summary1=summary1, summary2=summary2, article=article
            )
            system_prompt = COMPARISON_SYSTEM_PROMPT
        case "detection":
            system_prompt = DETECTION_SYSTEM_PROMPT
            prompt = DETECTION_PROMPT_TEMPLATE.format(
                summary1=summary1, summary2=summary2, article=article
            )
        case "detection_code":
            prompt = DETECTION_PROMPT_TEMPLATE_CODE.format(
                snippet1=summary1, snippet2=summary2, request=article
            )
            system_prompt = DETECTION_SYSTEM_PROMPT_CODE
        case "detection_vs_human":
            system_prompt = DETECTION_SYSTEM_PROMPT
            prompt = DETECTION_PROMPT_TEMPLATE_VS_HUMAN.format(
                summary1=summary1, summary2=summary2, article=article
            )
        case "detection_vs_model":
            system_prompt = DETECTION_SYSTEM_PROMPT
            prompt = DETECTION_PROMPT_TEMPLATE_VS_MODEL.format(
                summary1=summary1, summary2=summary2, article=article
            )            

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    attempts = 0
    while attempts < 10:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=history,
                max_tokens=10,
                temperature=0,
                logprobs=True if return_logprobs else None,
                top_logprobs=2 if return_logprobs else None,
            )
            if return_logprobs:
                return response.choices[0].logprobs.content
            else:
                return response.choices[0].message.content
        except openai.APITimeoutError:
            attempts += 1
            sleep(5)
            print(f"Timeout error after {attempts} attempts, retrying...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sleep(5)
            return "1"
    print(f"Failed after {attempts} attempts.")
    return "1"


def get_model_choice(
    summary1, summary2, article, choice_type, model, return_logprobs=False
):
    global openai_client
    apk = openai_api_key if "gpt" in model.lower() else gemini_api_key if "gemini" in model.lower() else lambda_api_key
    if openai_client.api_key != apk:
        print("Need new client")
        openai_client = init_client(model)

    if "claude" in model:
        return get_claude_choice(
            summary1, summary2, article, choice_type, model="claude-2.1"
        )
    if model.startswith("gpt"):
        return get_gpt_choice(summary1, summary2, article, choice_type, model=GPT_MODEL_ID.get(model, model), return_logprobs=return_logprobs)
    else:
        return get_gpt_choice(summary1, summary2, article, choice_type, model=model, return_logprobs=return_logprobs)


def get_gpt_choice_logprobs_with_sources(
    summary1, summary2, source1, source2, article, model
) -> dict:
    prompt = COMPARISON_PROMPT_TEMPLATE_WITH_SOURCES.format(
        summary1=summary1, summary2=summary2, source1=source1, source2=source2, article=article
    )
    system_prompt = COMPARISON_SYSTEM_PROMPT

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    attempts = 0
    while attempts < 10:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=history,
                max_tokens=10,
                temperature=0,
                logprobs=True,
                top_logprobs=1,
            )
            return response.choices[0].logprobs.content
        except openai.APITimeoutError:
            attempts += 1
            sleep(5)
            print(f"Timeout error after {attempts} attempts, retrying...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sleep(5)
            return "1"
    print(f"Failed after {attempts} attempts.")
    return "1"


def get_logprobs_choice_with_sources(
    summary1, summary2, source1, source2, article, model
):
    if "claude" in model:
        return get_claude_choice(
            summary1, summary2, article, source1, source2
        )
    if model.startswith("gpt"):
        return get_gpt_choice_logprobs_with_sources(summary1, summary2, source1, source2, article, model=GPT_MODEL_ID[model])
    else:
        return get_gpt_choice_logprobs_with_sources(summary1, summary2, source1, source2, article, model=model)


def get_gpt_recognition_logprobs(summary, article, model) -> dict:
    prompt = RECOGNITION_PROMPT_TEMPLATE.format(summary=summary, article=article)
    system_prompt = RECOGNITION_SYSTEM_PROMPT

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    attempts = 0
    while attempts < 10:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=history,
                max_tokens=10,
                temperature=0,
                logprobs=True,
                top_logprobs=1,
            )
            return response.choices[0].logprobs.content
        except openai.APITimeoutError:
            attempts += 1
            sleep(5)
            print(f"Timeout error after {attempts} attempts, retrying...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sleep(5)
            return "No"
    print(f"Failed after {attempts} attempts.")
    return "No"


def get_gpt_score(summary, article, model):
    system_prompt = SCORING_SYSTEM_PROMPT

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summary:\n{summary}\n\nArticle:\n{article}"},
    ]
    attempts = 0
    while attempts < 10:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=history,
                max_tokens=10,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
            )
            return response.choices[0].logprobs.content
        except openai.APITimeoutError:
            attempts += 1
            sleep(5)
            print(f"Timeout error after {attempts} attempts, retrying...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sleep(5)
            return "1"
    print(f"Failed after {attempts} attempts.")
    return "1"

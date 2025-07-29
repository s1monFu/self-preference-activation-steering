from transformers import pipeline
from utils.env_util import get_hf_token

def llama_summarizer():
    return pipeline(
        task="text-generation",
        model="meta-llama/Llama-3.1-8B-Instruct",
        token=get_hf_token(),
        trust_remote_code=True,
        max_new_tokens=128,
    )

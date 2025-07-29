import os
from dotenv import load_dotenv

def get_hf_token():
    load_dotenv()
    return os.getenv("HF_TOKEN")

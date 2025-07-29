import json, uuid
from utils.path_util import cnn_articles_file, summary_file_for
from utils.json_util import load_jsonl, write_jsonl
from utils.prompts import SUMMARIZE_PROMPT_TEMPLATE_CNN as prompt_template
from utils.summarizer_util import llama_summarizer

def write_single_summary():
    first_article = load_jsonl(cnn_articles_file())[0]
    prompt = prompt_template.format(article=first_article["text"])
    model = llama_summarizer()
    raw_output = model(prompt)[0]["generated_text"]
    record = {"key": first_article["key"], "summary": raw_output.strip()}
    write_jsonl([record], summary_file_for("llama3.1-8b-instruct"))

if __name__ == "__main__":
    write_single_summary()
    print("summarize one")

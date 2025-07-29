import uuid
from datasets import load_dataset
from utils.path_util import cnn_articles_file,check_if_data_set_exists
from utils.json_util import write_jsonl
from utils.env_util import get_hf_token

def write_cnn_articles():
    check_if_data_set_exists()
    dataset=load_dataset("cnn_dailymail","3.0.0",split="train")
    dataset = dataset.select(range(10000)) 
    records=[]
    for row in dataset:
        record={"key":str(uuid.uuid4()),"text":row["article"]}
        records.append(record)
    write_jsonl(records,cnn_articles_file())

if __name__ == "__main__":
    get_hf_token()
    write_cnn_articles()
    print("CNN articles written successfully.")

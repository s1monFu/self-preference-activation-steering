from pathlib import Path

def cnn_articles_file():
    directory=Path("../datasets/articles")
    directory.mkdir(parents=True,exist_ok=True)
    return directory/"cnn_articles.jsonl"

def check_if_data_set_exists():
    file_path=cnn_articles_file()
    if file_path.exists():
        print("Already exists")
        raise FileExistsError

import json
from pathlib import Path

def write_jsonl(records, file_path):
    file_handle = Path(file_path).open("w", encoding="utf-8")
    for record in records:
        json_line = json.dumps(record, ensure_ascii=False) + "\n"
        file_handle.write(json_line)
    file_handle.close()

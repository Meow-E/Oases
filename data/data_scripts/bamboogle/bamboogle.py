from datasets import load_dataset
import json

ds = load_dataset("chiayewken/bamboogle")

with open("bamboogle.jsonl", "w", encoding="utf-8") as f:
    for idx, example in enumerate(ds['test']):

        item = {
            "task_id": idx,
            "Question": example["Question"],
            "Answer": example["Answer"],
        }

        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + "\n")

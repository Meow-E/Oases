from datasets import load_dataset
import json

ds = load_dataset("google-research-datasets/nq_open")

with open("nq_train.jsonl", "w", encoding="utf-8") as f:
    for idx, example in enumerate(ds['train']):

        item = {
            "task_id": idx,
            "Question": example["question"],
            "Answer": " ".join(example["answer"]),
        }

        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + "\n")


with open("nq.jsonl", "w", encoding="utf-8") as f:
    for idx, example in enumerate(ds['validation']):

        item = {
            "task_id": idx,
            "Question": example["question"],
            "Answer": " ".join(example["answer"]),
        }

        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + "\n")
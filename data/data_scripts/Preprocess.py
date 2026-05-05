import os
import json
import argparse
from typing import Any, Dict, List

from tqdm import tqdm
from datasets import Dataset, concatenate_datasets


PROMPT = """You are an intelligent Research Agent designed to answer questions by retrieving information from the internet.

## Operational Logic
1.  Plan: Analyze the user's question. If the answer requires multiple steps, break it down.
2.  Search: Use the `<search>` tag to query the search engine. Keep queries specific and efficient.
3.  Observe: Wait for the `<result>` output provided by the system.
4.  Iterate: Analyze the search results in a `<think>` block.
    - If information is missing, formulate a new search query based on what you found.
    - If the information is sufficient, proceed to answer.
5.  Answer: Output the final answer in an `<answer>` tag. Keep it concise and direct.

## Response Format
- Thinking Process: <think> ... </think>
- Search Action: <search> ... </search>
- Final Answer: <answer> ... </answer>

## Constraints
- Stop after Search: Do not generate `<result>` tags yourself. Wait for the system.
- Search Limit: There is a maximum limit on the number of search rounds. If you receive a system warning that the limit is reached, you must immediately STOP searching, derive the best possible conclusion from the available info, and output the final answer.

## Example

Question: The director of the movie "Oppenheimer" was born in which city?

<think> I need to answer this in two steps. First, I must identify the director of the 2023 movie "Oppenheimer". Second, I need to find the birth city of that specific director. </think>
<search> director of movie Oppenheimer </search>
<result> The film Oppenheimer was written and directed by Christopher Nolan. </result>
<think> The director is Christopher Nolan. Now I need to search for his birth city. </think>
<search> Christopher Nolan birth city </search>
<result> Christopher Edward Nolan was born on July 30, 1970, in London, England. </result>
<think> The search results confirm he was born in London. I have the final answer. </think>
<answer> London </answer>"""


DATA_DICT = {
    "2wiki": {
        "data_source": "voidful/2WikiMultihopQA",
        "train_file": ["2wiki_train.jsonl"],
        "test_file": ["2wiki.jsonl"],
        "ability": "QA",
    },
    "bamboogle": {
        "data_source": "chiayewken/bamboogle",
        "test_file": ["bamboogle.jsonl"],
        "ability": "QA",
    },
    "hotpotqa": {
        "data_source": "hotpotqa/hotpot_qa",
        "train_file": ["hotpotqa_train.jsonl"],
        "test_file": ["hotpotqa.jsonl"],
        "ability": "QA",
    },
    "musique": {
        "data_source": "bdsaglam/musique",
        "train_file": ["musique_train.jsonl"],
        "test_file": ["musique.jsonl"],
        "ability": "QA",
    },
    "nq": {
        "data_source": "google-research-datasets/nq_open",
        "train_file": ["nq_train.jsonl"],
        "test_file": ["nq.jsonl"],
        "ability": "QA",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare QA datasets into veRL-style parquet files."
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Choose train or test split.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Dataset names to process. "
            "Example: --datasets hotpotqa or --datasets hotpotqa nq musique. "
            "Use --datasets all to process all datasets."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the processed dataset.",
    )

    parser.add_argument(
        "--save_jsonl",
        action="store_true",
        help="Also save the processed dataset as JSONL.",
    )

    return parser.parse_args()


def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def process_dataset(
    dataset_list: List[Dict[str, Any]],
    data_source: str,
    ability: str,
    split: str,
) -> Dataset:
    processed = []

    for idx, example in enumerate(dataset_list):
        question = example.get("Question", "")
        answer = example.get("Answer", "")

        question = "" if question is None else str(question)
        answer = "" if answer is None else str(answer)

        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": question},
            ],
            "ability": ability,
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": question,
            },
        }

        processed.append(data)

    return Dataset.from_list(processed)


def resolve_datasets(dataset_args: List[str]) -> List[str]:
    if "all" in dataset_args:
        return list(DATA_DICT.keys())

    invalid = [name for name in dataset_args if name not in DATA_DICT]
    if invalid:
        valid = ", ".join(DATA_DICT.keys())
        raise ValueError(f"Invalid dataset name(s): {invalid}. Valid datasets are: {valid}")

    return dataset_args


def main() -> None:
    args = parse_args()

    dataset_names = resolve_datasets(args.datasets)
    split = args.split
    file_key = f"{split}_file"

    os.makedirs(args.output_dir, exist_ok=True)

    all_datasets = []

    for dataset_name in tqdm(dataset_names, desc=f"Processing {split} datasets"):
        config = DATA_DICT[dataset_name]

        if file_key not in config:
            raise ValueError(
                f"Dataset '{dataset_name}' does not have a '{split}' split. "
                f"Available keys: {list(config.keys())}"
            )

        dataset_path = dataset_name
        raw_data = []

        for filename in config[file_key]:
            full_path = os.path.join(dataset_path, filename)

            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Input file not found: {full_path}")

            raw_data.extend(load_jsonl_file(full_path))

        if not raw_data:
            raise ValueError(f"No data loaded from dataset '{dataset_name}'.")

        processed_dataset = process_dataset(
            dataset_list=raw_data,
            data_source=config["data_source"],
            ability=config["ability"],
            split=split,
        )

        all_datasets.append(processed_dataset)

    if len(all_datasets) == 1:
        final_dataset = all_datasets[0]
    else:
        final_dataset = concatenate_datasets(all_datasets)

    if len(dataset_names) == 1:
        dataset_part = dataset_names[0]
    elif len(dataset_names) == len(DATA_DICT):
        dataset_part = "all"
    else:
        dataset_part = "_".join(dataset_names)

    output_prefix = f"{split}_{dataset_part}"
    parquet_path = os.path.join(args.output_dir, f"{output_prefix}.parquet")

    final_dataset.to_parquet(parquet_path)
    print(f"Successfully saved parquet to: {parquet_path}")
    print(f"Total samples: {len(final_dataset)}")

    if args.save_jsonl:
        jsonl_path = os.path.join(args.output_dir, f"{output_prefix}.jsonl")
        final_dataset.to_json(jsonl_path, force_ascii=False, lines=True)
        print(f"Successfully saved jsonl to: {jsonl_path}")


if __name__ == "__main__":
    main()

# python Preprocess.py \
#   --split train \
#   --datasets hotpotqa nq musique 2wiki \
#   --output_dir combined_outputs


# python Preprocess.py \
#   --split test \
#   --datasets all \
#   --output_dir combined_outputs
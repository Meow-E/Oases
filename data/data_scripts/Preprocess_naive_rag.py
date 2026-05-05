import os
import json
import time
import random
import requests
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from datasets import Dataset, concatenate_datasets


# =========================
# Hard-coded configs
# =========================
SEARCH_HOST = "10.158.150.20"
SEARCH_PORTS = ["8000", "8001", "8002", "8003", "8004", "8005", "8006", "8007"]
SEARCH_MAX_RESULTS = 3
SEARCH_TIMEOUT = 60
SEARCH_MAX_ATTEMPTS = 10

NUM_WORKERS = 64

OUTPUT_DIR = "combined_outputs"
TARGET_DATASETS = ["2wiki", "bamboogle", "hotpotqa", "musique", "nq"]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

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

RAG_SYSTEM_PROMPT = """You are a helpful question answering assistant.
Answer the question using the retrieved context.
If the context is insufficient, still provide your best possible answer.
Output only the final answer inside <answer> </answer> tags."""

RAG_USER_TEMPLATE = """Retrieved Context:
{context}

Question:
{question}"""


# =========================
# Search tool
# =========================
class SearchTool:
    def __init__(
        self,
        host: str,
        ports: List[str],
        max_results: int = 3,
        max_attempts: int = 10,
    ):
        self.headers = {"Content-Type": "application/json"}
        self.api_urls = [f"http://{host}:{port}/search" for port in ports]
        self.max_results = max_results
        self.max_attempts = max_attempts

    def execute(self, query: str, timeout: int = 60) -> List[str]:
        query = query.replace('"', "")
        payload = {"questions": [query], "N": self.max_results}

        attempts = 0
        tried_urls = set()

        while attempts < self.max_attempts:
            available_urls = [url for url in self.api_urls if url not in tried_urls]
            if not available_urls:
                available_urls = self.api_urls

            api_url = random.choice(available_urls)
            tried_urls.add(api_url)

            try:
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=timeout,
                )
                response.raise_for_status()

                data = response.json()
                if not data or "top_k_docs" not in data[0]:
                    raise ValueError(f"Unexpected API response: {data}")

                docs = data[0]["top_k_docs"]
                if not isinstance(docs, list):
                    raise ValueError(f"top_k_docs is not a list: {type(docs)}")

                return docs[: self.max_results]

            except (requests.exceptions.RequestException, ValueError) as e:
                attempts += 1
                print(f"⚠️ Search failed ({attempts}/{self.max_attempts}) at {api_url}: {e}")
                time.sleep(1)

        raise RuntimeError(f"❌ Search failed after {self.max_attempts} attempts. Query: {query}")


# =========================
# Utils
# =========================
def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_context_from_docs(docs: List[str]) -> str:
    if not docs:
        return ""

    parts = []
    for i, doc in enumerate(docs, start=1):
        doc = doc.strip()
        parts.append(f"[Document {i}]\n{doc}")
    return "\n\n".join(parts)


def retrieve_one(
    idx: int,
    example: Dict[str, Any],
    search_tool: SearchTool,
) -> Tuple[int, str, str, List[str], str]:
    question = example.get("Question", "")
    solution = example.get("Answer", "")

    try:
        retrieved_docs = search_tool.execute(question, timeout=SEARCH_TIMEOUT)
    except Exception as e:
        print(f"❌ Retrieval error at idx={idx}, question={question[:100]}..., error={e}")
        retrieved_docs = []

    retrieved_context = build_context_from_docs(retrieved_docs)
    return idx, question, solution, retrieved_docs, retrieved_context


def process_dataset_with_rag_parallel(
    dataset_list: List[Dict[str, Any]],
    data_source: str,
    ability: str,
    split: str,
    search_tool: SearchTool,
    num_workers: int = 32,
) -> Dataset:
    processed = [None] * len(dataset_list)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(retrieve_one, idx, example, search_tool)
            for idx, example in enumerate(dataset_list)
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Retrieving {data_source}",
        ):
            idx, question, solution, retrieved_docs, retrieved_context = future.result()

            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": RAG_USER_TEMPLATE.format(
                            context=retrieved_context,
                            question=question,
                        ),
                    },
                ],
                "ability": ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question,
                    "retrieved_docs": retrieved_docs,
                    "retrieved_context": retrieved_context,
                },
            }

            processed[idx] = data

    processed = [x for x in processed if x is not None]
    return Dataset.from_list(processed)


# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    search_tool = SearchTool(
        host=SEARCH_HOST,
        ports=SEARCH_PORTS,
        max_results=SEARCH_MAX_RESULTS,
        max_attempts=SEARCH_MAX_ATTEMPTS,
    )

    all_qa_test_datasets = []

    for dataset_name, config in tqdm(DATA_DICT.items(), desc="Processing datasets"):
        if dataset_name not in TARGET_DATASETS:
            continue

        dataset_path = dataset_name
        data_source = config["data_source"]
        ability = config["ability"]

        if "test_file" not in config:
            continue

        test_data = []
        for filename in config["test_file"]:
            full_path = os.path.join(dataset_path, filename)
            if not os.path.exists(full_path):
                print(f"⚠️ File not found, skip: {full_path}")
                continue
            test_data.extend(load_jsonl_file(full_path))

        if not test_data:
            print(f"⚠️ No test data found for {dataset_name}")
            continue

        current_test_ds = process_dataset_with_rag_parallel(
            dataset_list=test_data,
            data_source=data_source,
            ability=ability,
            split="test",
            search_tool=search_tool,
            num_workers=NUM_WORKERS,
        )
        all_qa_test_datasets.append(current_test_ds)

    if all_qa_test_datasets:
        print(f"Combining {len(all_qa_test_datasets)} QA datasets...")
        combined_dataset = concatenate_datasets(all_qa_test_datasets)

        parquet_path = os.path.join(OUTPUT_DIR, "test_naive_rag_qa.parquet")
        jsonl_path = os.path.join(OUTPUT_DIR, "test_naive_rag_qa.jsonl")

        combined_dataset.to_parquet(parquet_path)
        combined_dataset.to_json(jsonl_path, force_ascii=False, lines=True)

        print(f"✅ Saved parquet to: {parquet_path}")
        print(f"✅ Saved jsonl   to: {jsonl_path}")
        print(f"✅ Total samples : {len(combined_dataset)}")
    else:
        print("⚠️ No dataset was processed.")
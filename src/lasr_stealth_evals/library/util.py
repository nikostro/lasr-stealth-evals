import json
from typing import Any
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent.parent.parent / "data" / "dataset.jsonl"

def load_dataset(path: str = str(DATASET_PATH)) -> list[dict[str, Any]]:
    f = open(path, "r")
    dataset = [json.loads(line) for line in list(f)]
    f.close()
    return dataset

print(load_dataset())
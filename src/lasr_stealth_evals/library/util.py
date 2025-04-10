import json
from typing import Any
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"


def load_dataset(path: str = "dataset.json") -> dict[str, Any]:
    f = open(DATA_PATH / path, "r")
    dataset = json.load(f)
    f.close()
    return dataset

import json
from pathlib import Path
from pprint import pprint

def inspect_metadata():
    # Get the absolute path to the workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent
    
    # Load the JSON file
    json_path = workspace_root / 'logs_json' / 'sample.json'
    print(f"Looking for JSON file at: {json_path}")
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get the first sample's metadata
    first_sample = data["samples"][0]
    metadata = first_sample["metadata"]
    
    print("\nMetadata keys:")
    print("-" * 50)
    pprint(list(metadata.keys()))
    
    print("\nFull metadata structure:")
    print("-" * 50)
    pprint(metadata)

if __name__ == "__main__":
    inspect_metadata() 
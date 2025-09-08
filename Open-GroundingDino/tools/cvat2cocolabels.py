import json
import sys
from pathlib import Path

def create_label_map(input_path, output_path):
    # Read the label tree from file
    with open(input_path, "r") as f:
        label_tree = json.load(f)

    # Create label map: {"0": "box", "1": "home console", ...}
    label_map = {str(i): item["name"] for i, item in enumerate(label_tree)}

    # Save label map to file
    with open(output_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Label map saved to {output_path}")
    print(json.dumps(label_map, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_label_map.py <input_label_tree.json> <output_label_map.json>")
    else:
        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
        create_label_map(input_path, output_path)

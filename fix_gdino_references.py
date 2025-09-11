#!/usr/bin/env python3
"""
Fix GDINO JSON files to ensure consistent reference IDs and names from items.json

This script:
1. Loads items.json to create an ID-to-name mapping
2. Processes all JSON files in gdinoOutput/final
3. Fixes inconsistent 'id' fields in gdino_improved sections
4. Uses vector_reference_id as the proper ID when available
5. Updates reference_name to match items.json
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

# Paths
ITEMS_JSON_PATH = "items.json"
GDINO_FINAL_DIR = "gdinoOutput/final"

def load_items_mapping() -> Dict[str, str]:
    """Load items.json and create ID to name mapping."""
    print("Loading items.json...")
    
    with open(ITEMS_JSON_PATH, 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    # Create mapping: id -> name
    mapping = {}
    for item in items:
        item_id = item.get('id')
        item_name = item.get('name')
        if item_id and item_name:
            mapping[item_id] = item_name
    
    print(f"Loaded {len(mapping)} items from items.json")
    return mapping

def fix_gdino_improved_entry(entry: Dict[str, Any], items_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Fix a single gdino_improved entry."""
    if not isinstance(entry, dict):
        return entry
    
    # Check if vector_reference_id exists - this should become the main id
    if 'vector_reference_id' in entry:
        correct_id = entry['vector_reference_id']
        print(f"  → Fixing ID from '{entry.get('id', 'unknown')}' to '{correct_id}'")
        entry['id'] = correct_id
        
        # Remove vector_reference_id since it's now the main id
        entry.pop('vector_reference_id', None)
    
    # Update reference_name from items.json if we have the correct ID
    current_id = entry.get('id')
    if current_id and current_id in items_mapping:
        correct_name = items_mapping[current_id]
        current_name = entry.get('reference_name', '')
        
        if current_name != correct_name:
            print(f"  → Updating reference_name from '{current_name}' to '{correct_name}'")
            entry['reference_name'] = correct_name
    else:
        if current_id:
            print(f"  → Warning: ID '{current_id}' not found in items.json")
    
    return entry

def process_json_file(file_path: Path, items_mapping: Dict[str, str]) -> bool:
    """Process a single JSON file and fix gdino_improved entries."""
    try:
        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if gdino_improved exists
        if 'gdino_improved' not in data:
            return False
        
        changes_made = False
        gdino_improved = data['gdino_improved']
        
        # Process each entry in gdino_improved
        for key, entry in gdino_improved.items():
            if isinstance(entry, dict):
                original_entry = entry.copy()
                fixed_entry = fix_gdino_improved_entry(entry, items_mapping)
                
                if fixed_entry != original_entry:
                    changes_made = True
                    gdino_improved[key] = fixed_entry
        
        # Save the file if changes were made
        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        
        return False
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main execution function."""
    # Load items mapping
    items_mapping = load_items_mapping()
    
    # Find all JSON files in gdinoOutput/final
    gdino_final_path = Path(GDINO_FINAL_DIR)
    if not gdino_final_path.exists():
        print(f"Error: {GDINO_FINAL_DIR} directory not found!")
        return
    
    json_files = list(gdino_final_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    files_changed = 0
    for json_file in json_files:
        print(f"Processing: {json_file.relative_to(gdino_final_path)}")
        
        if process_json_file(json_file, items_mapping):
            files_changed += 1
            print(f"  ✓ Updated {json_file.relative_to(gdino_final_path)}")
        else:
            print(f"  - No changes needed")
    
    print(f"\nProcessing complete:")
    print(f"  Total files processed: {len(json_files)}")
    print(f"  Files changed: {files_changed}")
    print(f"  Files unchanged: {len(json_files) - files_changed}")

if __name__ == "__main__":
    main()
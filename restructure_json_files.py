#!/usr/bin/env python3
"""
Script to restructure JSON files to contain only gdino, gdino_readable, gdino_tokens, and enhanced_classification.
Removes url, bbox, and results sections.
"""

import json
import os
from pathlib import Path

def generate_readable_name(enhanced_classification, tokens):
    """Generate a readable name from enhanced_classification or tokens."""
    if enhanced_classification and "item_name" in enhanced_classification:
        item_name = enhanced_classification["item_name"]
        if "model" in enhanced_classification and enhanced_classification["model"]:
            return enhanced_classification["model"]
        elif item_name:
            # If we have DS Lite, make it Nintendo DS Lite Console
            if "DS Lite" in item_name:
                return "Nintendo DS Lite Console"
            elif "Switch" in item_name:
                return f"Nintendo {item_name} Console"
            else:
                return f"{item_name} Console"
    
    # Fallback to generating from tokens
    if "nintendo" in tokens and "switch" in tokens:
        return "Nintendo Switch Console"
    elif "nintendo" in tokens and "ds" in tokens and "lite" in tokens:
        return "Nintendo DS Lite Console"
    elif "nintendo" in tokens and "ds" in tokens:
        return "Nintendo DS Console"
    elif "playstation" in tokens:
        return "PlayStation Console"
    elif "xbox" in tokens:
        return "Xbox Console"
    else:
        return "Gaming Console"

def restructure_json_data(data):
    """Restructure JSON data to keep only the specified sections."""
    new_data = {}
    
    # Extract data from the nested structure
    for key, value in data.items():
        if isinstance(value, dict) and "tokens" in value:
            # Get enhanced_classification if it exists
            enhanced_classification = value.get("enhanced_classification", {})
            tokens = value.get("tokens", [])
            
            # Create the restructured format
            gdino_key = key
            
            # For gdino - use "2" as the default value as shown in the example
            new_data["gdino"] = {gdino_key: "2"}
            
            # For gdino_readable - generate from enhanced_classification or tokens
            readable_name = generate_readable_name(enhanced_classification, tokens)
            new_data["gdino_readable"] = {gdino_key: readable_name}
            
            # For gdino_tokens - use the existing tokens
            new_data["gdino_tokens"] = {gdino_key: tokens}
            
            # Add enhanced_classification if it exists
            if enhanced_classification:
                new_data["enhanced_classification"] = enhanced_classification
            
            break  # Only process the first valid entry
    
    return new_data

def process_json_files(directory_path):
    """Process all JSON files in the directory and subdirectories."""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist")
        return
    
    json_files = list(directory.rglob("*.json"))
    # Filter out backup files
    json_files = [f for f in json_files if not f.name.endswith('.backup')]
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restructure the data
            restructured_data = restructure_json_data(data)
            
            # Write back to the file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(restructured_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Processed: {json_file}")
            
        except Exception as e:
            print(f"✗ Error processing {json_file}: {e}")

if __name__ == "__main__":
    # Process files in gdinoOutput/final directory
    final_directory = "/Users/jovitaeliana/Personal/strustore/gdinoOutput/final"
    process_json_files(final_directory)
    print("Processing complete!")
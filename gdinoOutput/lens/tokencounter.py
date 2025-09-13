#!/usr/bin/env python3
"""
Token Counter Script for GDINO Lens Output

This script reads all JSON files from the lens subdirectories (1, 2, 3, 4, 5),
extracts all tokens, and counts their frequency of occurrence.
"""

import json
import os
from collections import Counter
from pathlib import Path
import csv


def extract_tokens_from_json(file_path):
    """
    Extract tokens from a single JSON file.
    
    Args:
        file_path (str): Path to the JSON file
    
    Returns:
        list: List of tokens found in the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract tokens from the JSON structure
        tokens = []
        
        # Check if it's a list (multiple entries)
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and 'tokens' in entry:
                    tokens.extend(entry['tokens'])
        
        # Check if it's a single dict with tokens
        elif isinstance(data, dict):
            if 'tokens' in data:
                tokens.extend(data['tokens'])
            
            # Also check nested structures
            for key, value in data.items():
                if isinstance(value, dict) and 'tokens' in value:
                    tokens.extend(value['tokens'])
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'tokens' in item:
                            tokens.extend(item['tokens'])
        
        return tokens
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def collect_all_tokens(base_dir):
    """
    Collect all tokens from lens directories 1-5.
    
    Args:
        base_dir (str): Base directory path (should be the lens directory)
    
    Returns:
        Counter: Counter object with token frequencies
    """
    all_tokens = []
    total_files = 0
    
    # Process directories 1, 2, 3, 4, 5
    for dir_num in range(1, 6):
        dir_path = Path(base_dir) / str(dir_num)
        
        if not dir_path.exists():
            print(f"Directory {dir_path} does not exist, skipping...")
            continue
        
        print(f"Processing directory: {dir_path}")
        files_in_dir = 0
        
        # Process all JSON files in the directory
        for json_file in dir_path.glob("*.json"):
            tokens = extract_tokens_from_json(json_file)
            all_tokens.extend(tokens)
            files_in_dir += 1
        
        print(f"  - Processed {files_in_dir} files")
        total_files += files_in_dir
    
    print(f"\nTotal files processed: {total_files}")
    print(f"Total tokens collected: {len(all_tokens)}")
    
    # Count token frequencies
    token_counter = Counter(all_tokens)
    print(f"Unique tokens found: {len(token_counter)}")
    
    return token_counter


def save_token_frequencies(token_counter, output_file):
    """
    Save token frequencies to a CSV file.
    
    Args:
        token_counter (Counter): Token frequency counter
        output_file (str): Output file path
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Token', 'Frequency'])
        
        # Sort by frequency (descending)
        for token, freq in token_counter.most_common():
            writer.writerow([token, freq])
    
    print(f"Token frequencies saved to: {output_file}")


def display_top_tokens(token_counter, top_n=50):
    """
    Display top N most frequent tokens.
    
    Args:
        token_counter (Counter): Token frequency counter
        top_n (int): Number of top tokens to display
    """
    print(f"\nTop {top_n} most frequent tokens:")
    print("-" * 40)
    
    for i, (token, freq) in enumerate(token_counter.most_common(top_n), 1):
        print(f"{i:2d}. {token:<25} {freq:>6} times")


def main():
    """Main function to run the token counter."""
    
    # Get the current directory (should be the lens directory)
    base_dir = Path(__file__).parent
    
    print("Token Counter for GDINO Lens Output")
    print("=" * 50)
    print(f"Base directory: {base_dir}")
    
    # Collect all tokens
    token_counter = collect_all_tokens(base_dir)
    
    if not token_counter:
        print("No tokens found!")
        return
    
    # Display statistics
    print(f"\nToken Statistics:")
    print(f"- Total unique tokens: {len(token_counter):,}")
    print(f"- Total token instances: {sum(token_counter.values()):,}")
    print(f"- Average frequency per token: {sum(token_counter.values()) / len(token_counter):.2f}")
    
    # Display top tokens
    display_top_tokens(token_counter, 50)
    
    # Save to CSV file
    output_file = base_dir / "token_frequencies.csv"
    save_token_frequencies(token_counter, output_file)
    
    # Additional analysis
    print(f"\nFrequency distribution:")
    freq_dist = Counter(token_counter.values())
    for freq, count in sorted(freq_dist.items(), reverse=True)[:10]:
        print(f"  {count} tokens appear {freq} time(s)")
    
    # Find tokens that appear only once
    singleton_count = sum(1 for freq in token_counter.values() if freq == 1)
    print(f"\nTokens appearing only once: {singleton_count} ({singleton_count/len(token_counter)*100:.1f}%)")
    
    # Find gaming-related tokens
    gaming_keywords = {
        'nintendo', 'playstation', 'xbox', 'switch', 'controller', 'console',
        'ds', 'dsi', 'wii', 'gameboy', 'gaming', 'handheld', 'ps1', 'ps2', 
        'ps3', 'ps4', 'ps5', '3ds', 'gba', 'n64', 'snes', 'gamecube'
    }
    
    gaming_tokens = {token: freq for token, freq in token_counter.items() 
                    if any(keyword in token.lower() for keyword in gaming_keywords)}
    
    if gaming_tokens:
        print(f"\nTop gaming-related tokens:")
        for token, freq in sorted(gaming_tokens.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {token:<20} {freq:>4} times")


if __name__ == "__main__":
    main()
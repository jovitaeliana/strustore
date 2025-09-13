#!/usr/bin/env python3
"""Simple test of query enhancement logic without model loading."""

import json
from pathlib import Path

def enhance_query(query: str, config: dict) -> str:
    """Simple version of query enhancement logic."""
    query_lower = query.lower()
    expanded_terms = set()

    gaming_synonyms = config.get('gaming_synonyms', {})
    token_mappings = config.get('token_mappings', {})

    # Apply direct token mappings first
    for token in query_lower.split():
        if token in token_mappings:
            expanded_terms.add(token_mappings[token])

    # Add specific matched synonyms
    for synonym_category in gaming_synonyms.values():
        if isinstance(synonym_category, dict):
            for _, terms in synonym_category.items():
                for term in terms:
                    if term.lower() in query_lower:
                        expanded_terms.add(terms[0].lower())  # Add canonical form
                        break

    # Add brand context selectively
    brands = gaming_synonyms.get('brands', {})
    for brand_name, brand_terms in brands.items():
        if any(brand_term.lower() in query_lower for brand_term in brand_terms):
            expanded_terms.add(brand_name.lower())
            break

    # Create enhanced query
    query_parts = [query]
    if expanded_terms:
        expanded_terms.discard(query_lower)
        query_parts.extend(sorted(list(expanded_terms))[:5])

    return ' | '.join(query_parts)

def main():
    # Load config
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    test_queries = [
        "Nintendo DS",
        "DS Lite",
        "PlayStation", 
        "controller",
        "con"
    ]
    
    print("Query Enhancement Test Results:")
    print("=" * 50)
    
    for query in test_queries:
        enhanced = enhance_query(query, config)
        print(f"'{query}' → '{enhanced}'")
    
    print("\nExpected Improvements:")
    print("- 'Nintendo DS' should expand with DS synonyms")
    print("- 'DS Lite' should add nintendo ds lite canonical form")  
    print("- 'controller' should remain as is")
    print("- 'con' should expand to 'controller'")

if __name__ == "__main__":
    main()
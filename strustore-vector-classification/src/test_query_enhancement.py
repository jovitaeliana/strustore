#!/usr/bin/env python3
"""Test query enhancement matching item contextual text approach."""

from create_vector_database import VectorDatabaseSearcher
import json

def test_query_enhancement():
    """Test that queries are enhanced to match item embeddings."""
    
    print("Testing Query Enhancement...")
    
    # Create searcher instance
    searcher = VectorDatabaseSearcher(
        model_path='intfloat/multilingual-e5-base',
        database_path='../models/vector_database'
    )
    
    # Test query enhancement
    test_queries = [
        "Nintendo DS",
        "DS Lite", 
        "PlayStation",
        "controller",
        "コントローラー"
    ]
    
    print("Query Enhancement Results:")
    print("=" * 60)
    
    for query in test_queries:
        enhanced = searcher._enhance_query(query)
        print(f"Original: {query}")
        print(f"Enhanced: {enhanced}")
        print()
    
    print("Item Contextual Text Examples:")
    print("=" * 60)
    
    # Show what item contextual text looks like for comparison
    with open('/Users/jovitaeliana/Personal/strustore/models/vector_database/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Find DS Lite item
    for item in metadata:
        if 'DS Lite' in item['name'] and item['name'] == 'DS Lite':
            print(f"Item: {item['name']}")
            print(f"Context: {item['contextual_text']}")
            print()
            break
    
    # Find controller item
    for item in metadata:
        if 'GC Con' in item['name']:
            print(f"Item: {item['name']}")
            print(f"Context: {item['contextual_text']}")
            print()
            break
            
    print("Now the query and item embeddings should be in the same semantic space!")

if __name__ == "__main__":
    test_query_enhancement()
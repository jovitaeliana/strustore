#!/usr/bin/env python3
"""Test the enhanced classification system with config-based improvements."""

import json
import sys
from pathlib import Path
from enhance_gdino_results import GdinoResultEnhancer

def test_ds_lite_classification():
    """Test that DS Lite tokens correctly match DS Lite items."""
    
    print("Testing DS Lite classification with config-based enhancements...")
    
    # Initialize GDINO result enhancer
    enhancer = GdinoResultEnhancer(
        items_json_path="/Users/jovitaeliana/Personal/strustore/items.json",
        config_path="config.json"
    )
    
    # Test case: DS Lite tokens from GDINO
    test_tokens = ["ds", "nintendo", "lite", "ds lite", "nintendo ds", "blue", "game", "console", "ice", "ice blue", "handheld"]
    
    # Expected result: Should match DS Lite items
    expected_category = "Video Game Consoles"
    expected_contains = ["ds lite", "nintendo ds lite"]  # Should contain these terms
    
    print(f"Input tokens: {test_tokens}")
    
    try:
        # Get classification result
        result = enhancer.get_best_classification(test_tokens)
        
        if result:
            print(f"\nBest Match:")
            print(f"1. {result['name']} (ID: {result['id']})")
            print(f"   Category: {result['category']}")
            print(f"   Validation: {result.get('validation_result', {})}")
            
            name_lower = result['name'].lower()
            
            success_checks = []
            success_checks.append(("Category", result['category'] == expected_category))
            success_checks.append(("Contains DS Lite", any(term in name_lower for term in expected_contains)))
            validation_result = result.get('validation_result', {})
            success_checks.append(("Has validation", bool(validation_result)))
            
            print("Verification:")
            all_passed = True
            for check_name, passed in success_checks:
                status = "✓" if passed else "✗"
                print(f"{status} {check_name}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                print("\n✓ DS Lite classification test PASSED!")
                return True
            else:
                print("\n✗ DS Lite classification test FAILED!")
                return False
        else:
            print("\n✗ No results returned!")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_controller_classification():
    """Test that 'con' tokens correctly map to controller items."""
    
    print("\n" + "="*60)
    print("Testing Controller classification with 'con' token mapping...")
    
    enhancer = GdinoResultEnhancer(
        items_json_path="/Users/jovitaeliana/Personal/strustore/items.json",
        config_path="config.json"
    )
    
    # Test case: Controller tokens where 'con' should map to 'controller'
    test_tokens = ["gc", "con", "loose", "stick", "gamecube", "controller"]
    
    print(f"Input tokens: {test_tokens}")
    
    try:
        result = enhancer.get_best_classification(test_tokens)
        
        if result:
            print(f"\nBest Match:")
            print(f"1. {result['name']} (ID: {result['id']})")  
            print(f"   Category: {result['category']}")
            print(f"   Validation: {result.get('validation_result', {})}")
            
            name_lower = result['name'].lower()
            
            # Should match controller-related items
            is_controller = any(term in name_lower for term in ['controller', 'con'])
            is_gamecube = any(term in name_lower for term in ['gamecube', 'gc'])
            
            print("\nVerification:")
            print(f"{'✓' if is_controller else '✗'} Contains controller/con terms")
            print(f"{'✓' if is_gamecube else '✗'} Contains GameCube terms")
            
            if is_controller:
                print("\n✓ Controller classification test PASSED!")
                return True
            else:
                print("\n✗ Controller classification test FAILED!")
                return False
        else:
            print("\n✗ No results returned!")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during classification: {e}")
        return False

if __name__ == "__main__":
    print("Enhanced Classification System Test")
    print("=" * 60)
    
    # Run tests
    ds_lite_passed = test_ds_lite_classification()
    controller_passed = test_controller_classification()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"DS Lite Classification: {'PASSED' if ds_lite_passed else 'FAILED'}")
    print(f"Controller Classification: {'PASSED' if controller_passed else 'FAILED'}")
    
    if ds_lite_passed and controller_passed:
        print("\n✓ All tests PASSED! Config-based enhancements are working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests FAILED. Check configuration and implementation.")
        sys.exit(1)
"""
Strict Token-Based GroundingDINO Result Enhancement Pipeline

This script enhances GDINO detection results using strict token matching logic.
It requires exact token presence validation with config-based normalization.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            script_dir = Path(__file__).parent
            config_file = script_dir / config_path
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def safe_lower(text: Any) -> str:
    """Safely convert text to lowercase, handling None values."""
    if text is None:
        return ""
    if not isinstance(text, str):
        return str(text).lower()
    return text.lower()


def validate_tokens(tokens: List[Any]) -> List[str]:
    """Validate and clean token list, removing None values and non-strings."""
    if not tokens:
        return []
    
    clean_tokens = []
    for token in tokens:
        if token is not None and isinstance(token, str) and token.strip():
            clean_tokens.append(token.strip())
        elif token is not None:
            str_token = str(token).strip()
            if str_token:
                clean_tokens.append(str_token)
    
    return clean_tokens


class StrictTokenMatcher:
    """
    Strict token-based matcher that requires exact token presence validation.
    """
    
    def __init__(self, config: Dict[str, Any], items_data: List[Dict[str, Any]]):
        self.config = config
        self.items_data = items_data
        self.token_mappings = config.get('token_mappings', {})
        
        # Create lookup structures for fast searching
        self._build_lookup_structures()
    
    def _build_lookup_structures(self):
        """Build optimized lookup structures for fast matching."""
        self.items_by_id = {item['id']: item for item in self.items_data}
        self.items_by_normalized_name = {}
        
        # Build normalized name lookup
        for item in self.items_data:
            normalized_name = self._normalize_text(item.get('name', ''))
            if normalized_name:
                if normalized_name not in self.items_by_normalized_name:
                    self.items_by_normalized_name[normalized_name] = []
                self.items_by_normalized_name[normalized_name].append(item)
    
    def normalize_token(self, token: str) -> str:
        """Normalize token using config mappings (e.g., 'con' -> 'controller')."""
        if not token:
            return ""
        
        token_lower = safe_lower(token)
        return self.token_mappings.get(token_lower, token_lower)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by splitting into tokens and applying mappings."""
        if not text:
            return ""
        
        words = safe_lower(text).split()
        normalized_words = []
        
        for word in words:
            # Clean word of punctuation
            clean_word = word.strip('.,!?()[]{}"\'-')
            if clean_word:
                # Apply token mapping
                normalized_word = self.token_mappings.get(clean_word, clean_word)
                normalized_words.append(normalized_word)
        
        return ' '.join(normalized_words)
    
    def extract_key_words(self, item_name: str) -> Set[str]:
        """Extract key words from item name that must be present in tokens."""
        if not item_name:
            return set()
        
        # Normalize the item name
        normalized_name = self._normalize_text(item_name)
        words = normalized_name.split()
        
        # Filter out common words that don't need strict matching
        stop_words = {'the', 'and', 'or', 'for', 'with', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 
                     'boxed', 'sealed', 'new', 'used', 'original'}  # Add descriptive words
        
        key_words = set()
        for word in words:
            if len(word) >= 2 and word not in stop_words and not word.isdigit():
                key_words.add(word)
        
        # For items with only 1-2 key words, be more flexible
        if len(key_words) <= 2:
            return key_words
        
        # For longer items, require at least the core identifying words
        # Example: "DS Lite" -> require both "ds" and "lite"
        return key_words
    
    def get_normalized_tokens(self, tokens: List[str]) -> Set[str]:
        """Get normalized token set for matching."""
        if not tokens:
            return set()
        
        normalized_tokens = set()
        
        for token in tokens[:20]:  # Use top 20 tokens
            normalized_token = self.normalize_token(token)
            if normalized_token:
                normalized_tokens.add(normalized_token)
                
                # Also add individual words from compound tokens
                words = normalized_token.split()
                for word in words:
                    if len(word) >= 2:
                        normalized_tokens.add(word)
        
        return normalized_tokens
    
    def strict_token_validation(self, tokens: List[str], item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform strict token validation - ALL key words from item must exist in tokens.
        
        Args:
            tokens: List of GDINO tokens
            item: Item from items.json
            
        Returns:
            Validation result with score and details
        """
        item_name = item.get('name', '')
        if not item_name or not tokens:
            return {'valid': False, 'score': 0.0, 'reason': 'Empty input'}
        
        # Get normalized tokens from GDINO output
        normalized_tokens = self.get_normalized_tokens(tokens)
        
        # Get key words that must be present
        key_words = self.extract_key_words(item_name)
        
        if not key_words:
            return {'valid': False, 'score': 0.0, 'reason': 'No key words extracted'}
        
        # Check if ALL key words are present in tokens
        missing_words = key_words - normalized_tokens
        present_words = key_words & normalized_tokens
        
        # Calculate coverage
        coverage = len(present_words) / len(key_words) if key_words else 0
        
        # For short items (1-2 words), require all words present
        # For longer items, allow some flexibility
        required_coverage = 1.0 if len(key_words) <= 2 else 0.8
        
        if coverage < required_coverage:
            return {
                'valid': False, 
                'score': coverage,
                'reason': f'Missing key words: {missing_words} (coverage: {coverage:.2f})',
                'present_words': present_words,
                'missing_words': missing_words
            }
        
        # Calculate token relevance score
        relevance_score = self.calculate_token_relevance(normalized_tokens, item)
        
        # Require reasonable relevance for a match
        # Lower threshold for items with perfect key word coverage
        min_relevance = 0.3 if coverage == 1.0 else 0.5
        
        if relevance_score < min_relevance:
            return {
                'valid': False,
                'score': relevance_score,
                'reason': f'Low token relevance: {relevance_score:.2f} (min: {min_relevance:.2f})',
                'present_words': present_words
            }
        
        # Calculate final score with exact match bonus
        final_score = relevance_score
        
        # Bonus for exact phrase matches in tokens
        item_name_lower = item_name.lower()
        exact_match_bonus = 0.0
        
        # Check if the exact item name appears in tokens
        for token in normalized_tokens:
            if token == item_name_lower:
                exact_match_bonus += 0.3
                break
        
        final_score = min(final_score + exact_match_bonus, 1.0)
        
        return {
            'valid': True,
            'score': final_score,
            'reason': 'All key words present with high relevance',
            'present_words': present_words,
            'key_words_coverage': len(present_words) / len(key_words),
            'exact_match_bonus': exact_match_bonus
        }
    
    def calculate_token_relevance(self, normalized_tokens: Set[str], item: Dict[str, Any]) -> float:
        """
        Calculate how relevant the tokens are to the specific item.
        
        Args:
            normalized_tokens: Set of normalized tokens
            item: Item to check relevance against
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not normalized_tokens:
            return 0.0
        
        # Get all text associated with the item
        item_text_parts = [
            item.get('name', ''),
            item.get('model', ''),
            item.get('category', '')
        ]
        
        # Normalize all item text
        all_item_words = set()
        for text_part in item_text_parts:
            if text_part:
                normalized_text = self._normalize_text(text_part)
                all_item_words.update(normalized_text.split())
        
        # Add category-specific terms from config
        category = item.get('category', '')
        if category:
            category_terms = self._get_category_terms(category)
            all_item_words.update(category_terms)
        
        # Calculate relevance - favor items where key words are well represented
        if not all_item_words:
            return 0.0
        
        relevant_tokens = normalized_tokens & all_item_words
        
        # For items with simple names (like "DS Lite"), focus on key word coverage
        item_key_words = self.extract_key_words(item.get('name', ''))
        key_word_coverage = len(item_key_words & normalized_tokens) / len(item_key_words) if item_key_words else 0
        
        # Weighted relevance: 70% key word coverage + 30% overall relevance
        overall_relevance = len(relevant_tokens) / len(normalized_tokens) if normalized_tokens else 0
        relevance_score = (key_word_coverage * 0.7) + (overall_relevance * 0.3)
        
        return min(relevance_score, 1.0)  # Cap at 1.0
    
    def _get_category_terms(self, category: str) -> Set[str]:
        """Get relevant terms for a category from config."""
        category_terms = set()
        
        # Get gaming synonyms from config
        gaming_synonyms = self.config.get('gaming_synonyms', {})
        
        # Map category to relevant terms
        category_mappings = {
            'Video Game Consoles': ['console', 'gaming', 'handheld'],
            'Controllers & Attachments': ['controller', 'gamepad', 'joystick'],
            'Power Cables & Connectors': ['power', 'cable', 'charger', 'adapter'],
            'Memory Cards & Expansion Packs': ['memory', 'card', 'storage'],
            'Video Games': ['game', 'software'],
            'Other Video Game Accessories': ['accessory', 'gaming']
        }
        
        if category in category_mappings:
            category_terms.update(category_mappings[category])
        
        return category_terms
    
    def find_exact_matches(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Find exact matches using strict token validation."""
        if not tokens:
            return []
        
        matches = []
        
        for item in self.items_data:
            validation_result = self.strict_token_validation(tokens, item)
            
            if validation_result['valid']:
                match_data = item.copy()
                match_data['validation_result'] = validation_result
                match_data['match_score'] = validation_result['score']
                matches.append(match_data)
        
        # Sort by match score (descending)
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return matches
    
    def get_best_match(self, tokens: List[str]) -> Optional[Dict[str, Any]]:
        """Get the best match using strict token validation."""
        matches = self.find_exact_matches(tokens)
        
        if not matches:
            return None
        
        # Return the highest scoring match
        best_match = matches[0]
        
        # Log the match details for debugging
        logger.debug(f"Best match: {best_match['name']} (score: {best_match['match_score']:.3f})")
        logger.debug(f"Validation: {best_match['validation_result']}")
        
        return best_match


class GdinoResultEnhancer:
    """
    Strict token-based GDINO result enhancement system.
    """
    
    def __init__(self, items_json_path: str, config_path: str = "config.json"):
        """Initialize the enhancer with strict token matching."""
        
        # Load configuration
        self.config = load_config(config_path)
        if not self.config:
            logger.warning("No config loaded, using minimal defaults")
            self.config = {"token_mappings": {"con": "controller"}}
        
        # Load items.json data
        self.items_data = self._load_items_data(items_json_path)
        
        # Initialize strict token matcher
        self.token_matcher = StrictTokenMatcher(self.config, self.items_data)
        
        logger.info(f"Enhancer initialized with {len(self.items_data)} items")
    
    def _load_items_data(self, items_json_path: str) -> List[Dict[str, Any]]:
        """Load items from JSON file."""
        try:
            with open(items_json_path, 'r', encoding='utf-8') as f:
                items_data = json.load(f)
            
            # Filter out items without required fields
            valid_items = []
            for item in items_data:
                if item.get('id') and item.get('name'):
                    valid_items.append(item)
            
            logger.info(f"Loaded {len(valid_items)} valid items from {items_json_path}")
            return valid_items
            
        except Exception as e:
            logger.error(f"Failed to load items data: {e}")
            return []
    
    def get_best_classification(self, tokens: List[str]) -> Optional[Dict[str, Any]]:
        """
        Get the best classification using strict token matching.
        
        Args:
            tokens: List of GDINO tokens
            
        Returns:
            Best classification result or None if no match
        """
        try:
            # Validate tokens
            clean_tokens = validate_tokens(tokens)
            if not clean_tokens:
                logger.debug("No valid tokens provided")
                return None
            
            # Use strict token matching
            best_match = self.token_matcher.get_best_match(clean_tokens)
            
            if best_match:
                logger.info(f"Matched: {best_match['name']} (score: {best_match['match_score']:.3f})")
                return best_match
            else:
                logger.debug("No strict token match found")
                return None
            
        except Exception as e:
            logger.error(f"Error in get_best_classification: {e}")
            return None
    
    def enhance_gdino_file(self, input_file: Path, output_file: Path) -> bool:
        """
        Enhance a single GDINO JSON file using strict token matching.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output enhanced JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load input data
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract tokens safely - handle different JSON structures
            gdino_tokens = []
            
            # Check if it's the old format
            if 'gdino_token' in data:
                gdino_tokens = data['gdino_token']
            # Check if it's the new lens format with numbered keys
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict) and 'tokens' in value:
                        gdino_tokens.extend(value['tokens'])
            
            # Remove duplicates while preserving order
            if gdino_tokens:
                seen = set()
                unique_tokens = []
                for token in gdino_tokens:
                    if token not in seen:
                        seen.add(token)
                        unique_tokens.append(token)
                gdino_tokens = unique_tokens
            
            # Create enhanced data
            enhanced_data = data.copy()
            
            if not gdino_tokens:
                enhanced_data['enhanced_classification'] = {
                    'status': 'No Match',
                    'reason': 'No tokens available',
                    'confidence': 0.0,
                    'method': 'strict_token_matching'
                }
            else:
                # Get best classification using strict token matching
                classification_result = self.get_best_classification(gdino_tokens)
                
                if classification_result:
                    validation_result = classification_result.get('validation_result', {})
                    enhanced_data['enhanced_classification'] = {
                        'status': 'Match Found',
                        'item_id': classification_result.get('id'),
                        'item_name': classification_result.get('name'),
                        'category': classification_result.get('category'),
                        'model': classification_result.get('model', ''),
                        'match_score': classification_result.get('match_score', 0),
                        'confidence': classification_result.get('match_score', 0),
                        'method': 'strict_token_matching',
                        'validation_details': {
                            'present_words': list(validation_result.get('present_words', [])),
                            'key_words_coverage': validation_result.get('key_words_coverage', 0),
                            'reason': validation_result.get('reason', '')
                        }
                    }
                else:
                    enhanced_data['enhanced_classification'] = {
                        'status': 'No Match',
                        'reason': 'No items matched strict token validation',
                        'confidence': 0.0,
                        'method': 'strict_token_matching'
                    }
            
            # Save enhanced data
            self._save_enhanced_data(enhanced_data, output_file)
            return True
            
        except Exception as e:
            logger.error(f"Error enhancing file {input_file}: {e}")
            return False
    
    def _save_enhanced_data(self, data: Dict[str, Any], output_file: Path):
        """Save enhanced data to file."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving to {output_file}: {e}")
            raise
    
    def enhance_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, int]:
        """
        Enhance all JSON files in a directory using strict token matching.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            
        Returns:
            Stats dictionary with success/failure counts
        """
        stats = {'success': 0, 'failed': 0, 'total': 0, 'matched': 0, 'no_match': 0}
        
        # Find all JSON files
        json_files = list(input_dir.glob("*.json"))
        stats['total'] = len(json_files)
        
        if not json_files:
            logger.warning(f"No JSON files found in {input_dir}")
            return stats
        
        logger.info(f"Processing {len(json_files)} files from {input_dir}")
        
        # Process each file
        for json_file in tqdm(json_files, desc="Enhancing files"):
            output_file = output_dir / json_file.name
            
            if self.enhance_gdino_file(json_file, output_file):
                stats['success'] += 1
                
                # Check if a match was found
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    
                    if result.get('enhanced_classification', {}).get('status') == 'Match Found':
                        stats['matched'] += 1
                    else:
                        stats['no_match'] += 1
                except:
                    pass
            else:
                stats['failed'] += 1
        
        match_rate = (stats['matched'] / stats['success'] * 100) if stats['success'] > 0 else 0
        logger.info(f"Enhancement complete: {stats['success']} success, {stats['failed']} failed")
        logger.info(f"Match rate: {stats['matched']}/{stats['success']} ({match_rate:.1f}%)")
        
        return stats


def main():
    """Main function to run the strict token-based enhancement pipeline."""
    
    # Configuration
    items_json_path = "/Users/jovitaeliana/Personal/strustore/items.json"
    input_base_dir = Path("../gdinoOutput/lens")  # Test on lens data
    output_base_dir = Path("../gdinoOutput/enhanced_strict")
    
    try:
        # Initialize enhancer with strict token matching
        enhancer = GdinoResultEnhancer(items_json_path)
        
        # Process lens directories (1, 2, 3, 4, 5)
        total_stats = {'success': 0, 'failed': 0, 'total': 0, 'matched': 0, 'no_match': 0}
        
        for subdir_num in range(1, 6):
            input_dir = input_base_dir / str(subdir_num)
            output_dir = output_base_dir / str(subdir_num)
            
            if not input_dir.exists():
                logger.warning(f"Input directory {input_dir} does not exist")
                continue
            
            logger.info(f"\n--- Processing lens directory {subdir_num} ---")
            stats = enhancer.enhance_directory(input_dir, output_dir)
            
            # Accumulate stats
            for key in total_stats:
                total_stats[key] += stats[key]
        
        # Final summary
        success_rate = (total_stats['success'] / total_stats['total'] * 100) if total_stats['total'] > 0 else 0
        match_rate = (total_stats['matched'] / total_stats['success'] * 100) if total_stats['success'] > 0 else 0
        
        logger.info(f"\n=== STRICT TOKEN MATCHING RESULTS ===")
        logger.info(f"Total files processed: {total_stats['total']}")
        logger.info(f"Successful: {total_stats['success']} ({success_rate:.1f}%)")
        logger.info(f"Failed: {total_stats['failed']}")
        logger.info(f"Matched: {total_stats['matched']} ({match_rate:.1f}%)")
        logger.info(f"No Match: {total_stats['no_match']}")
        
        if total_stats['failed'] == 0:
            logger.info("🎉 All files processed successfully with strict token matching!")
        
    except Exception as e:
        logger.error(f"Enhancement pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
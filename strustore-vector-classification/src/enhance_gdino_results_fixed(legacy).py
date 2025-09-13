"""
Simplified GroundingDINO Result Enhancement Pipeline

This script enhances existing GroundingDINO detection results using a simplified,
robust approach that focuses on accuracy over complexity. It fixes the critical
None value handling bug and removes over-engineered classification logic.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            # Try relative to script directory
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
            # Convert non-string tokens to string if they're not None
            str_token = str(token).strip()
            if str_token:
                clean_tokens.append(str_token)
    
    return clean_tokens


class SimplifiedCategoryDetector:
    """
    Simplified category detection using config-based gaming terms and direct similarity matching.
    Removes unnecessary complexity and focuses on accuracy.
    """
    
    def __init__(self, model: SentenceTransformer, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.gaming_terms = self._extract_gaming_terms()
        self.token_mappings = config.get('token_mappings', {})
        self.similarity_threshold = 0.6  # Simple threshold instead of complex boundaries
    
    def _extract_gaming_terms(self) -> set:
        """Extract all gaming terms from config for quick lookup."""
        gaming_terms = set()
        gaming_synonyms = self.config.get('gaming_synonyms', {})
        
        for category, subcategories in gaming_synonyms.items():
            if isinstance(subcategories, dict):
                for items in subcategories.values():
                    if isinstance(items, list):
                        gaming_terms.update(safe_lower(term) for term in items)
            elif isinstance(subcategories, list):
                gaming_terms.update(safe_lower(term) for term in subcategories)
        
        return gaming_terms
    
    def normalize_token(self, token: str) -> str:
        """Normalize token using config mappings (e.g., 'con' -> 'controller')."""
        if not token:
            return ""
        
        token_lower = safe_lower(token)
        return self.token_mappings.get(token_lower, token_lower)
    
    def detect_brand_from_tokens(self, tokens: List[str]) -> Optional[str]:
        """Detect dominant brand from token list."""
        if not tokens:
            return None
            
        brands = self.config.get('gaming_synonyms', {}).get('brands', {})
        brand_counts = {}
        
        for token in tokens:
            normalized_token = self.normalize_token(token)
            for brand, brand_terms in brands.items():
                if any(safe_lower(term) in normalized_token for term in brand_terms):
                    brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        if brand_counts:
            return max(brand_counts.items(), key=lambda x: x[1])[0]
        return None
    
    def is_gaming_related(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Simplified gaming detection using config-based terms.
        
        Args:
            tokens: List of tokens to analyze
            
        Returns:
            Dict with gaming detection results
        """
        if not tokens:
            return {'is_gaming': False, 'confidence': 0.0, 'gaming_tokens': []}
        
        # Validate and normalize tokens
        clean_tokens = validate_tokens(tokens)
        if not clean_tokens:
            return {'is_gaming': False, 'confidence': 0.0, 'gaming_tokens': []}
        
        # Normalize tokens using mappings
        normalized_tokens = [self.normalize_token(token) for token in clean_tokens[:20]]
        
        # Find gaming-related tokens
        gaming_tokens = []
        for token in normalized_tokens:
            if token in self.gaming_terms:
                gaming_tokens.append(token)
        
        # Calculate confidence based on gaming token ratio
        confidence = len(gaming_tokens) / len(normalized_tokens) if normalized_tokens else 0.0
        
        # Simple threshold for gaming classification
        is_gaming = confidence >= 0.2 and len(gaming_tokens) >= 1
        
        return {
            'is_gaming': is_gaming,
            'confidence': confidence,
            'gaming_tokens': gaming_tokens,
            'total_tokens': len(normalized_tokens)
        }
    
    def validate_brand_consistency(self, tokens: List[str], match_result: Dict[str, Any]) -> bool:
        """
        Validate that the detected brand matches the classification result.
        
        Args:
            tokens: Detected tokens
            match_result: Classification match result
            
        Returns:
            True if brands are consistent or no brand conflict detected
        """
        detected_brand = self.detect_brand_from_tokens(tokens)
        if not detected_brand:
            return True  # No strong brand signal
        
        match_name_lower = safe_lower(match_result.get('name', ''))
        
        # Check for brand conflicts
        brand_conflicts = {
            'nintendo': ['playstation', 'xbox'],
            'sony': ['nintendo', 'xbox'],
            'microsoft': ['nintendo', 'playstation']
        }
        
        if detected_brand in brand_conflicts:
            conflicting_brands = brand_conflicts[detected_brand]
            if any(brand in match_name_lower for brand in conflicting_brands):
                return False
        
        return True
    
    def calculate_token_coverage(self, tokens: List[str], target_text: str) -> float:
        """
        Calculate what percentage of tokens are covered in the target text.
        
        Args:
            tokens: Input tokens
            target_text: Text to check against
            
        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if not tokens or not target_text:
            return 0.0
        
        clean_tokens = validate_tokens(tokens)
        if not clean_tokens:
            return 0.0
        
        target_lower = safe_lower(target_text)
        covered_tokens = 0
        
        for token in clean_tokens[:10]:  # Check first 10 tokens
            normalized_token = self.normalize_token(token)
            if normalized_token in target_lower:
                covered_tokens += 1
        
        return covered_tokens / min(len(clean_tokens), 10)


class GdinoResultEnhancer:
    """
    Simplified GDINO result enhancement system with robust error handling.
    """
    
    def __init__(self, 
                 vector_db_path: str, 
                 model_name: str = "intfloat/multilingual-e5-base",
                 config_path: str = "config.json"):
        """Initialize the enhancer with simplified configuration."""
        
        # Load configuration
        self.config = load_config(config_path)
        if not self.config:
            logger.warning("No config loaded, using minimal defaults")
            self.config = {"token_mappings": {"con": "controller"}}
        
        self.vector_db_path = Path(vector_db_path)
        self.model_name = model_name
        
        # Initialize components
        self.model = None
        self.faiss_index = None
        self.metadata = None
        self.item_lookup = None
        self.category_detector = None
        
        # Load vector database
        self._load_vector_database()
        
        # Initialize simplified category detector
        if self.model:
            self.category_detector = SimplifiedCategoryDetector(self.model, self.config)
    
    def _load_vector_database(self):
        """Load the vector database components."""
        try:
            logger.info(f"Loading vector database from: {self.vector_db_path}")
            
            # Load model
            self.model = SentenceTransformer(self.model_name)
            
            # Load FAISS index
            faiss_path = self.vector_db_path / "faiss_index.bin"
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            # Load metadata
            metadata_path = self.vector_db_path / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Load lookup table
            lookup_path = self.vector_db_path / "item_lookup.json"
            with open(lookup_path, 'r', encoding='utf-8') as f:
                self.item_lookup = json.load(f)
            
            logger.info(f"Vector database loaded: {len(self.metadata)} items ready")
            
        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            raise
    
    def get_best_classification(self, tokens: List[str], top_k: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get the best classification for given tokens with robust validation.
        
        Args:
            tokens: List of tokens to classify
            top_k: Number of top results to consider
            
        Returns:
            Best classification result or None if no good match
        """
        try:
            # Validate tokens first
            clean_tokens = validate_tokens(tokens)
            if not clean_tokens:
                logger.debug("No valid tokens provided")
                return None
            
            # Check if gaming-related
            if self.category_detector:
                gaming_check = self.category_detector.is_gaming_related(clean_tokens)
                if not gaming_check['is_gaming']:
                    logger.debug(f"Tokens not gaming-related: {gaming_check}")
                    return None
            
            # Create query text with normalized tokens
            if self.category_detector:
                normalized_tokens = [self.category_detector.normalize_token(token) for token in clean_tokens[:10]]
            else:
                normalized_tokens = clean_tokens[:10]
            
            query_text = ' '.join(normalized_tokens)
            
            # Search vector database
            results = self._search_vector_database(query_text, top_k)
            if not results:
                return None
            
            # Validate and score results
            best_result = self._validate_and_score_results(clean_tokens, results)
            
            return best_result
            
        except Exception as e:
            logger.error(f"Error in get_best_classification: {e}")
            return None
    
    def _search_vector_database(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Search the vector database for similar items."""
        try:
            # Encode query
            query_embedding = self.model.encode([f"passage: {query_text}"])
            faiss.normalize_L2(query_embedding.astype('float32'))
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(score)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def _validate_and_score_results(self, tokens: List[str], results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate and score search results."""
        if not results:
            return None
        
        best_result = None
        best_score = 0.0
        
        for result in results:
            try:
                # Basic similarity threshold
                if result.get('similarity_score', 0) < self.similarity_threshold:
                    continue
                
                # Brand consistency check
                if self.category_detector and not self.category_detector.validate_brand_consistency(tokens, result):
                    logger.debug(f"Brand conflict for {result.get('name', '')}")
                    continue
                
                # Token coverage check
                if self.category_detector:
                    coverage = self.category_detector.calculate_token_coverage(tokens, result.get('name', ''))
                    if coverage < 0.2:  # Require at least 20% token coverage
                        logger.debug(f"Low token coverage ({coverage:.2f}) for {result.get('name', '')}")
                        continue
                
                # Calculate combined score
                similarity = result.get('similarity_score', 0)
                coverage = self.category_detector.calculate_token_coverage(tokens, result.get('name', '')) if self.category_detector else 0.5
                combined_score = similarity * 0.7 + coverage * 0.3
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = result.copy()
                    best_result['combined_score'] = combined_score
                    best_result['token_coverage'] = coverage
                
            except Exception as e:
                logger.error(f"Error validating result {result.get('name', '')}: {e}")
                continue
        
        return best_result
    
    def enhance_gdino_file(self, input_file: Path, output_file: Path) -> bool:
        """
        Enhance a single GDINO JSON file.
        
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
            
            # Extract tokens safely
            gdino_tokens = data.get('gdino_token', [])
            if not gdino_tokens:
                logger.debug(f"No tokens found in {input_file}")
                # Still save the file with "No Match" classification
                enhanced_data = data.copy()
                enhanced_data['enhanced_classification'] = {
                    'status': 'No Match',
                    'reason': 'No tokens available',
                    'confidence': 0.0
                }
                self._save_enhanced_data(enhanced_data, output_file)
                return True
            
            # Get best classification
            classification_result = self.get_best_classification(gdino_tokens)
            
            # Create enhanced data
            enhanced_data = data.copy()
            
            if classification_result:
                enhanced_data['enhanced_classification'] = {
                    'status': 'Match Found',
                    'item_id': classification_result.get('id'),
                    'item_name': classification_result.get('name'),
                    'category': classification_result.get('category'),
                    'similarity_score': classification_result.get('similarity_score', 0),
                    'combined_score': classification_result.get('combined_score', 0),
                    'token_coverage': classification_result.get('token_coverage', 0),
                    'confidence': classification_result.get('combined_score', 0)
                }
            else:
                enhanced_data['enhanced_classification'] = {
                    'status': 'No Match',
                    'reason': 'No suitable classification found',
                    'confidence': 0.0
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
        Enhance all JSON files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            
        Returns:
            Stats dictionary with success/failure counts
        """
        stats = {'success': 0, 'failed': 0, 'total': 0}
        
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
            else:
                stats['failed'] += 1
        
        logger.info(f"Enhancement complete: {stats['success']} success, {stats['failed']} failed")
        return stats


def main():
    """Main function to run the enhancement pipeline."""
    
    # Configuration
    vector_db_path = "models/vector_database"
    input_base_dir = Path("gdinoOutput/final")
    output_base_dir = Path("gdinoOutput/enhanced")
    
    try:
        # Initialize enhancer
        enhancer = GdinoResultEnhancer(vector_db_path)
        
        # Process all subdirectories (1, 2, 3, 4, 5)
        total_stats = {'success': 0, 'failed': 0, 'total': 0}
        
        for subdir_num in range(1, 6):
            input_dir = input_base_dir / str(subdir_num)
            output_dir = output_base_dir / str(subdir_num)
            
            if not input_dir.exists():
                logger.warning(f"Input directory {input_dir} does not exist")
                continue
            
            logger.info(f"\n--- Processing directory {subdir_num} ---")
            stats = enhancer.enhance_directory(input_dir, output_dir)
            
            # Accumulate stats
            for key in total_stats:
                total_stats[key] += stats[key]
        
        # Final summary
        success_rate = (total_stats['success'] / total_stats['total'] * 100) if total_stats['total'] > 0 else 0
        logger.info(f"\n=== FINAL SUMMARY ===")
        logger.info(f"Total files processed: {total_stats['total']}")
        logger.info(f"Successful: {total_stats['success']} ({success_rate:.1f}%)")
        logger.info(f"Failed: {total_stats['failed']}")
        
        if total_stats['failed'] == 0:
            logger.info("🎉 All files processed successfully!")
        else:
            logger.warning(f"⚠️  {total_stats['failed']} files failed processing")
        
    except Exception as e:
        logger.error(f"Enhancement pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
"""
GroundingDINO Result Enhancement Pipeline

This script enhances existing GroundingDINO detection results by using the vector database
for improved semantic classification. It processes all JSON files in gdinoOutput/final/
and adds vector database classifications while preserving original results for comparison.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GdinoResultEnhancer:
    """
    Enhances GroundingDINO detection results using vector database semantic search.
    """
    
    def __init__(self, 
                 vector_db_path: str = "models/vector_database",
                 model_path: str = "intfloat/multilingual-e5-base",
                 gdino_output_dir: str = "../gdinoOutput/final"):
        """
        Initialize the result enhancer.
        
        Args:
            vector_db_path: Path to vector database directory
            model_path: Path to trained semantic model
            gdino_output_dir: Path to GroundingDINO output directory
        """
        self.vector_db_path = Path(vector_db_path)
        self.model_path = model_path
        self.gdino_output_dir = Path(gdino_output_dir)
        
        # Initialize components
        self.model = None
        self.faiss_index = None
        self.metadata = None
        self.item_lookup = None
        
    def load_vector_database(self) -> None:
        """Load the vector database and model for semantic search."""
        try:
            logger.info(f"🔄 Loading vector database from: {self.vector_db_path}")
            
            # Load model
            logger.info(f"Loading model: {self.model_path}")
            self.model = SentenceTransformer(str(self.model_path))
            
            # Load FAISS index
            faiss_path = self.vector_db_path / "faiss_index.bin"
            if not faiss_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            # Load metadata
            metadata_path = self.vector_db_path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Load lookup table
            lookup_path = self.vector_db_path / "item_lookup.json"
            if not lookup_path.exists():
                raise FileNotFoundError(f"Item lookup not found: {lookup_path}")
            with open(lookup_path, 'r', encoding='utf-8') as f:
                self.item_lookup = json.load(f)
            
            logger.info(f"✅ Vector database loaded: {len(self.metadata)} items ready for search")
            
            # Display category statistics
            category_counts = {}
            for item in self.metadata:
                category = item.get('category', 'Unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            logger.info("Available categories:")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {category}: {count} items")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load vector database: {e}")
    
    def search_similar_items(self, tokens: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar items based on token array using semantic search.
        
        Args:
            tokens: List of detection tokens
            k: Number of similar items to return
            
        Returns:
            List of similar items with similarity scores
        """
        if not tokens:
            return []
        
        try:
            # Prioritize important gaming-related tokens
            important_tokens = []
            secondary_tokens = []
            
            # Gaming-specific keywords to prioritize
            gaming_keywords = {
                'nintendo', 'playstation', 'xbox', 'switch', 'controller', 'console',
                'ds', 'psp', 'vita', 'gamecube', 'wii', 'joy', 'con', 'joycon',
                'ps1', 'ps2', 'ps3', 'ps4', 'ps5', '3ds', 'gba', 'n64', 'snes',
                'scph', 'oled', 'pro', 'slim', 'lite', 'memory', 'card'
            }
            
            # Filter and prioritize tokens
            for token in tokens:
                token_clean = token.strip().lower()
                if len(token_clean) < 2:
                    continue
                    
                # Skip common non-gaming words
                if token_clean in {'de', 'en', 'com', 'por', 'para', 'con', 'las', 'los', 'el', 'la', 'w', 'r', 'l', 'buy', 'best', 'online', 'youtube', 'tiktok', 'facebook', 'mercadolibre', 'wallapop', 'amazon', 'ebay', 'price', 'precio', 'estado', 'condition', 'new', 'used', 'segunda', 'mano', 'original', 'genuine', 'oem', 'tested', 'working', 'bundle', 'set', 'kit', 'box', 'caja'}:
                    continue
                    
                # Prioritize gaming-related tokens
                if any(keyword in token_clean for keyword in gaming_keywords):
                    important_tokens.append(token)
                else:
                    secondary_tokens.append(token)
            
            # Combine tokens with priority to important ones
            all_tokens = important_tokens + secondary_tokens
            if not all_tokens:
                return []
            
            # Create search query - use up to 15 tokens with preference for important ones
            query_tokens = all_tokens[:15]
            query_text = ' '.join(query_tokens)
            
            # Add E5 query prefix for better retrieval performance
            query_text = f"query: {query_text}"
            
            # Encode query
            query_embedding = self.model.encode([query_text])
            faiss.normalize_L2(query_embedding.astype('float32'))
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score > 0.0:  # Valid result with positive similarity
                    item_meta = self.metadata[idx].copy()
                    item_meta['similarity_score'] = float(score)
                    results.append(item_meta)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for tokens: {e}")
            return []
    
    def get_best_classification(self, tokens: List[str], min_similarity: float = 0.3) -> Optional[Tuple[str, str, float]]:
        """
        Get the best classification for a set of tokens.
        
        Args:
            tokens: Detection tokens
            min_similarity: Minimum similarity threshold
            
        Returns:
            Tuple of (item_id, readable_name, similarity_score) or None if no good match
        """
        if not tokens:
            return None
        
        # Search for similar items
        similar_items = self.search_similar_items(tokens, k=3)
        
        if not similar_items:
            return None
        
        # Get the best match
        best_item = similar_items[0]
        similarity = best_item['similarity_score']
        
        # Only return if similarity is above threshold
        if similarity >= min_similarity:
            return (
                best_item['id'],
                best_item['name'], 
                similarity
            )
        
        return None
    
    def enhance_gdino_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Enhance a single GroundingDINO result file with vector database classifications.
        
        Args:
            file_path: Path to the GroundingDINO JSON file
            
        Returns:
            Enhanced JSON data
        """
        try:
            # Load original file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create enhanced data with proper order - gdino_tokens at the end
            enhanced_data = {}
            
            # Add fields in desired order
            enhanced_data['gdino'] = data.get('gdino', {})
            enhanced_data['gdino_readable'] = data.get('gdino_readable', {})
            
            # Initialize new enhancement fields
            enhanced_data['gdino_improved'] = {}
            enhanced_data['gdino_improved_readable'] = {}
            enhanced_data['gdino_similarity_scores'] = {}
            enhanced_data['gdino_classification_metadata'] = {
                'processed_date': str(Path(__file__).stat().st_mtime),
                'vector_db_version': '2.0',
                'model_used': str(self.model_path),
                'min_similarity_threshold': 0.3
            }
            
            # Add gdino_tokens at the end
            enhanced_data['gdino_tokens'] = data.get('gdino_tokens', {})
            
            # Process each detection
            gdino_tokens = data.get('gdino_tokens', {})
            
            for detection_id, tokens in gdino_tokens.items():
                if not tokens or not isinstance(tokens, list):
                    # No tokens available - mark as no match
                    enhanced_data['gdino_improved'][detection_id] = ""
                    enhanced_data['gdino_improved_readable'][detection_id] = "No Match"
                    enhanced_data['gdino_similarity_scores'][detection_id] = 0.0
                    continue
                
                # Get best classification from vector database
                classification_result = self.get_best_classification(tokens)
                
                if classification_result:
                    item_id, readable_name, similarity = classification_result
                    enhanced_data['gdino_improved'][detection_id] = item_id
                    enhanced_data['gdino_improved_readable'][detection_id] = readable_name
                    enhanced_data['gdino_similarity_scores'][detection_id] = round(similarity, 4)
                else:
                    # No good match found
                    enhanced_data['gdino_improved'][detection_id] = ""
                    enhanced_data['gdino_improved_readable'][detection_id] = "No Match"
                    enhanced_data['gdino_similarity_scores'][detection_id] = 0.0
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to enhance file {file_path}: {e}")
            return {}
    
    def process_all_files(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Process all GroundingDINO result files and enhance them with vector database classifications.
        
        Args:
            dry_run: If True, process files but don't write results (for testing)
            
        Returns:
            Processing summary statistics
        """
        try:
            logger.info(f"🔍 Scanning for GroundingDINO result files in: {self.gdino_output_dir}")
            
            # Find all JSON files in the directory structure
            json_files = list(self.gdino_output_dir.rglob("*.json"))
            
            if not json_files:
                logger.warning(f"No JSON files found in {self.gdino_output_dir}")
                return {'status': 'no_files_found'}
            
            logger.info(f"Found {len(json_files)} JSON files to process")
            
            # Processing statistics
            stats = {
                'total_files': len(json_files),
                'processed_files': 0,
                'failed_files': 0,
                'total_detections': 0,
                'enhanced_detections': 0,
                'no_match_detections': 0,
                'similarity_distribution': {'0.0-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, '0.9-1.0': 0},
                'category_improvements': {},
                'failed_file_paths': []
            }
            
            # Process files with progress bar
            with tqdm(json_files, desc="Enhancing GroundingDINO results") as pbar:
                for file_path in pbar:
                    try:
                        pbar.set_description(f"Processing {file_path.name}")
                        
                        # Enhance the file
                        enhanced_data = self.enhance_gdino_file(file_path)
                        
                        if not enhanced_data:
                            stats['failed_files'] += 1
                            stats['failed_file_paths'].append(str(file_path))
                            continue
                        
                        # Update statistics
                        gdino_improved = enhanced_data.get('gdino_improved', {})
                        similarity_scores = enhanced_data.get('gdino_similarity_scores', {})
                        
                        for detection_id, item_id in gdino_improved.items():
                            stats['total_detections'] += 1
                            
                            if item_id and item_id != "":
                                stats['enhanced_detections'] += 1
                                
                                # Track similarity distribution
                                similarity = similarity_scores.get(detection_id, 0.0)
                                if similarity >= 0.9:
                                    stats['similarity_distribution']['0.9-1.0'] += 1
                                elif similarity >= 0.7:
                                    stats['similarity_distribution']['0.7-0.9'] += 1
                                elif similarity >= 0.5:
                                    stats['similarity_distribution']['0.5-0.7'] += 1
                                elif similarity >= 0.3:
                                    stats['similarity_distribution']['0.3-0.5'] += 1
                                else:
                                    stats['similarity_distribution']['0.0-0.3'] += 1
                                
                                # Track category improvements
                                if item_id in self.item_lookup:
                                    category = self.item_lookup[item_id].get('category', 'Unknown')
                                    stats['category_improvements'][category] = stats['category_improvements'].get(category, 0) + 1
                                    
                            else:
                                stats['no_match_detections'] += 1
                        
                        # Write enhanced file (unless dry run)
                        if not dry_run:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(enhanced_data, f, indent=4, ensure_ascii=False)
                        
                        stats['processed_files'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                        stats['failed_files'] += 1
                        stats['failed_file_paths'].append(str(file_path))
            
            # Calculate enhancement rate
            enhancement_rate = (stats['enhanced_detections'] / stats['total_detections'] * 100) if stats['total_detections'] > 0 else 0
            
            # Log summary
            logger.info("\n" + "="*60)
            logger.info("🎯 GDINO RESULT ENHANCEMENT COMPLETE")
            logger.info("="*60)
            logger.info(f"📁 Files processed: {stats['processed_files']}/{stats['total_files']}")
            logger.info(f"🔍 Total detections: {stats['total_detections']}")
            logger.info(f"✅ Enhanced detections: {stats['enhanced_detections']} ({enhancement_rate:.1f}%)")
            logger.info(f"❌ No match detections: {stats['no_match_detections']}")
            
            if stats['failed_files'] > 0:
                logger.warning(f"⚠️  Failed files: {stats['failed_files']}")
            
            logger.info("\nSimilarity Score Distribution:")
            for range_name, count in stats['similarity_distribution'].items():
                percentage = (count / stats['enhanced_detections'] * 100) if stats['enhanced_detections'] > 0 else 0
                logger.info(f"  {range_name}: {count} ({percentage:.1f}%)")
            
            logger.info("\nTop Enhanced Categories:")
            sorted_categories = sorted(stats['category_improvements'].items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_categories[:10]:
                logger.info(f"  {category}: {count} detections")
            
            stats['enhancement_rate'] = round(enhancement_rate, 2)
            stats['status'] = 'success'
            
            return stats
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'status': 'error', 'error': str(e)}


def main():
    """Main function to enhance GroundingDINO results."""
    
    # Configuration
    vector_db_path = "models/vector_database"
    model_path = "intfloat/multilingual-e5-base" 
    gdino_output_dir = "../gdinoOutput/final"
    
    logger.info("🚀 Starting GroundingDINO Result Enhancement Pipeline")
    
    # Initialize enhancer
    enhancer = GdinoResultEnhancer(vector_db_path, model_path, gdino_output_dir)
    
    # Load vector database
    enhancer.load_vector_database()
    
    # Process all files
    logger.info("🔄 Processing all GroundingDINO result files...")
    results = enhancer.process_all_files(dry_run=False)
    
    if results['status'] == 'success':
        logger.info(f"✅ Enhancement completed successfully!")
        logger.info(f"📊 Enhanced {results['enhanced_detections']} out of {results['total_detections']} detections")
        logger.info(f"📈 Enhancement rate: {results['enhancement_rate']}%")
    else:
        logger.error(f"❌ Enhancement failed: {results.get('error', 'Unknown error')}")
    
    return results


if __name__ == '__main__':
    main()
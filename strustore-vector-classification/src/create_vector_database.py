"""
Vector Database Creation Script

This script creates the vector database from the trained model and master items list.
The vector database serves as the "reference library" for real-time classification.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import pickle
from datetime import datetime

# Core ML libraries
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Import position-weighted classification system
import sys
sys.path.append('/Users/jovitaeliana/Personal/strustore')
from position_weighted_embeddings import PositionWeightedTokenClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabaseCreator:
    """
    Creates and manages the vector database for gaming console classification.
    """
    
    def __init__(self, 
                 model_path: str,
                 master_items_path: str,
                 database_output_path: str = "models/vector_database"):
        """
        Initialize the vector database creator.
        
        Args:
            model_path: Path to trained semantic model
            master_items_path: Path to master items CSV file
            database_output_path: Output path for vector database
        """
        self.model_path = Path(model_path)
        self.master_items_path = Path(master_items_path)
        self.database_output_path = Path(database_output_path)
        
        # Create output directory
        self.database_output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.master_items = None
        self.vector_index = None
        self.item_metadata = {}
        
    def load_model(self) -> None:
        """Load the trained semantic model."""
        try:
            # Check if it's a HuggingFace model name or local path
            if str(self.model_path).startswith(('sentence-transformers/', 'intfloat/')):
                logger.info(f"Loading pre-trained model: {self.model_path}")
                self.model = SentenceTransformer(str(self.model_path))
            else:
                if not self.model_path.exists():
                    raise FileNotFoundError(f"Model not found: {self.model_path}")
                
                logger.info(f"Loading trained model from: {self.model_path}")
                self.model = SentenceTransformer(str(self.model_path))
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def load_master_items(self) -> None:
        """Load master items from JSON file."""
        try:
            if not self.master_items_path.exists():
                raise FileNotFoundError(f"Master items file not found: {self.master_items_path}")
            
            logger.info(f"Loading master items from: {self.master_items_path}")
            
            # Load JSON data
            with open(self.master_items_path, 'r', encoding='utf-8') as f:
                items_data = json.load(f)
            
            # Convert JSON to DataFrame format for compatibility
            processed_items = []
            for item in items_data:
                # Skip items without required fields
                if not item.get('id') or not item.get('name'):
                    continue
                    
                # Clean and validate item name
                item_name = str(item['name']).strip()
                if not item_name:
                    continue
                    
                processed_items.append({
                    'id': item['id'],
                    'item': item_name,  # Keep 'item' for compatibility
                    'name': item_name,   # Also store as 'name'
                    'category': item.get('category', 'Unknown'),
                    'model': item.get('model', ''),
                    'legacyId': item.get('legacyId')
                })
            
            # Create DataFrame
            self.master_items = pd.DataFrame(processed_items)
            
            # Validate we have data
            if len(self.master_items) == 0:
                raise ValueError("No valid items found in JSON file")
            
            logger.info(f"Loaded {len(self.master_items)} items from master list")
            
            # Display category distribution
            category_counts = self.master_items['category'].value_counts()
            logger.info("Category distribution:")
            for category, count in category_counts.head(10).items():
                logger.info(f"  {category}: {count} items")
            
            # Display sample items
            logger.info("Sample items:")
            for idx, row in self.master_items.head().iterrows():
                logger.info(f"  ID {row['id']}: {row['name']} ({row['category']})")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load master items: {e}")
    
    def create_contextual_text(self, item_name: str, category: str, model: str = "", item_id: str = "", position_weighted: bool = True) -> str:
        """
        Create richer contextual text for enhanced semantic embedding while preserving Firebase data integrity.
        
        Args:
            item_name: Original item name from Firebase
            category: Original Firebase category (exact string match)
            model: Original Firebase model field
            item_id: Original Firebase item ID
            
        Returns:
            Enhanced contextual text string in format: item_name | id: item_id | category: exact_category | expanded_synonyms
        """
        # Enhanced gaming-specific synonym expansion based on positive training pairs
        gaming_synonyms = {
            # Console abbreviations
            'PS2': 'PlayStation 2',
            'PS1': 'PlayStation 1', 
            'PS3': 'PlayStation 3',
            'PS4': 'PlayStation 4',
            'PS5': 'PlayStation 5',
            'DS': 'Nintendo DS',
            '3DS': 'Nintendo 3DS',
            'GBA': 'Game Boy Advance',
            'GBC': 'Game Boy Color',
            'GC': 'GameCube',
            'N64': 'Nintendo 64',
            'SNES': 'Super Nintendo Entertainment System',
            'NES': 'Nintendo Entertainment System',
            'PSP': 'PlayStation Portable',
            'Vita': 'PlayStation Vita',
            'Switch': 'Nintendo Switch',
            'Wii': 'Nintendo Wii',
            'Xbox': 'Microsoft Xbox',
            
            # Model codes from positive pairs
            'NTR-001': 'Nintendo DS Original Console',
            'USG-001': 'Nintendo DS Lite Console', 
            'TWL-001': 'Nintendo DSi Console',
            'CTR-001': 'Nintendo 3DS Console',
            'HDH-001': 'Nintendo Switch Lite Console',
            'AGS-001': 'Game Boy Advance Console',
            'SHVC-001': 'Super Nintendo Console',
            
            # Color/condition terms from training data
            'ホワイト': 'white',
            'ブラック': 'black',
            'ブルー': 'blue',
            'レッド': 'red',
            'ピンク': 'pink',
            'シルバー': 'silver',
            '本体': 'console',
            'コントローラー': 'controller',
            '任天堂': 'nintendo',
            'プレイステーション': 'playstation',
            '動作確認済み': 'tested working',
            '新品': 'new',
            '中古': 'used',
            '美品': 'mint condition'
        }
        
        # Start with original Firebase name (preserved exactly)
        context_parts = [item_name]
        
        # Add ID context if available
        if item_id and item_id.strip():
            context_parts.append(f"id: {item_id}")
        
        # Add category context with exact Firebase category string
        context_parts.append(f"category: {category}")
        
        # Add model context if available
        if model and model.strip():
            context_parts.append(f"model: {model}")
        
        # Gaming synonym expansion based on positive training pairs
        expanded_terms = []
        item_lower = item_name.lower()
        
        # Direct synonym matches
        for abbrev, full_name in gaming_synonyms.items():
            if abbrev.lower() in item_lower or abbrev in item_name:
                expanded_terms.append(full_name)
        
        # Console-specific contextual expansions
        if any(term in item_lower for term in ['(ps2)', 'ps2']):
            expanded_terms.extend(['PlayStation 2 console game', 'Sony PlayStation 2'])
        elif any(term in item_lower for term in ['(ds)', 'nintendo ds', 'ds lite']):
            expanded_terms.extend(['Nintendo DS handheld game', 'portable gaming device'])
        elif any(term in item_lower for term in ['(gc)', 'gamecube']):
            expanded_terms.extend(['GameCube console game', 'Nintendo GameCube'])
        elif any(term in item_lower for term in ['(n64)', 'nintendo 64']):
            expanded_terms.extend(['Nintendo 64 console game', 'N64 cartridge'])
        elif any(term in item_lower for term in ['(wii)', 'nintendo wii']):
            expanded_terms.extend(['Nintendo Wii console game', 'motion control gaming'])
        elif any(term in item_lower for term in ['(switch)', 'nintendo switch']):
            expanded_terms.extend(['Nintendo Switch console game', 'hybrid handheld console'])
        elif any(term in item_lower for term in ['(psp)', 'playstation portable']):
            expanded_terms.extend(['PlayStation Portable handheld game', 'Sony PSP'])
        elif any(term in item_lower for term in ['(gba)', 'game boy advance']):
            expanded_terms.extend(['Game Boy Advance handheld game', 'Nintendo GBA'])
        
        # Category-specific semantic expansions
        if category == 'Video Games':
            expanded_terms.append('video game software')
        elif category == 'Gaming Consoles':
            expanded_terms.extend(['gaming console hardware', 'video game system'])
        elif category == 'Controllers & Attachments':
            expanded_terms.extend(['gaming controller device', 'gamepad peripheral'])
        elif category == 'Handheld Consoles':
            expanded_terms.extend(['portable handheld gaming system', 'mobile gaming device'])
        elif category == 'Memory Cards & Expansion Packs':
            expanded_terms.extend(['memory storage device', 'game save storage'])
        elif category == 'Power Cables & Connectors':
            expanded_terms.extend(['power supply cable', 'electrical connector'])
        
        # Add brand context for better semantic matching
        if any(brand in item_lower for brand in ['nintendo', '任天堂']):
            expanded_terms.append('Nintendo brand gaming product')
        elif any(brand in item_lower for brand in ['playstation', 'sony', 'ps', 'プレイステーション']):
            expanded_terms.append('Sony PlayStation gaming product')
        elif any(brand in item_lower for brand in ['xbox', 'microsoft']):
            expanded_terms.append('Microsoft Xbox gaming product')
        elif any(brand in item_lower for brand in ['sega']):
            expanded_terms.append('Sega gaming product')
        
        # Combine all contextual information
        if expanded_terms:
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term in expanded_terms:
                if term.lower() not in seen:
                    seen.add(term.lower())
                    unique_terms.append(term)
            context_parts.extend(unique_terms)
        
        # Apply position-weighted enhancement if enabled
        if position_weighted:
            # Initialize position-weighted classifier for contextual enhancement
            classifier = PositionWeightedTokenClassifier()
            
            # Create hardware-focused context based on position weighting principles
            hardware_context = []
            item_lower = item_name.lower()
            
            # Priority hardware terms (equivalent to position 0-3 in GDINO ranking)
            priority_terms = []
            for term in classifier.hardware_terms:
                if term in item_lower:
                    priority_terms.append(term)
            
            # Add top priority terms first (simulating high-confidence GDINO tokens)
            if priority_terms:
                hardware_context.extend(priority_terms[:4])  # Top 4 most relevant
            
            # Add brand context with high weighting
            for brand, terms in classifier.brand_terms.items():
                if any(term in item_lower for term in terms):
                    hardware_context.append(f"{brand} gaming hardware")
                    break
            
            if hardware_context:
                context_parts.extend(hardware_context)
        
        return ' | '.join(context_parts)

    def generate_embeddings(self, use_position_weighting: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generate embeddings for all master items with enhanced contextual text.
        
        Returns:
            Tuple of (embeddings_array, metadata_list)
        """
        try:
            logger.info("🔄 Generating contextual embeddings for master items...")
            
            items = self.master_items['name'].tolist()
            item_ids = self.master_items['id'].tolist()
            item_categories = self.master_items['category'].tolist()
            item_models = self.master_items['model'].tolist()
            
            # Create contextual texts for better semantic understanding
            contextual_texts = []
            for item_name, item_id, category, model in zip(items, item_ids, item_categories, item_models):
                contextual_text = self.create_contextual_text(item_name, category, model, item_id, use_position_weighting)
                contextual_texts.append(contextual_text)
            
            logger.info(f"Created contextual texts for {len(contextual_texts)} items")
            
            # Generate embeddings in batches for efficiency
            batch_size = 32
            all_embeddings = []
            
            # For E5 models, add "passage:" prefix for better retrieval performance
            model_name = str(self.model_path)
            use_e5_prefix = 'e5' in model_name.lower()
            
            for i in range(0, len(contextual_texts), batch_size):
                batch_texts = contextual_texts[i:i + batch_size]
                
                # Add E5 passage prefix if using E5 model
                if use_e5_prefix:
                    batch_texts = [f"passage: {text}" for text in batch_texts]
                
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            
            # Create enhanced metadata for each item with position-weighting info
            metadata = []
            classifier = PositionWeightedTokenClassifier() if use_position_weighting else None
            
            for idx, (item_id, item_name, item_category, item_model, contextual_text) in enumerate(
                zip(item_ids, items, item_categories, item_models, contextual_texts)
            ):
                meta = {
                    'id': item_id,
                    'name': item_name,                    # Original Firebase name
                    'category': item_category,            # Original Firebase category
                    'model': item_model if item_model else '',  # Original Firebase model
                    'contextual_text': contextual_text,   # Enhanced context for embedding
                    'vector_index': idx,
                    'embedding_norm': float(np.linalg.norm(embeddings[idx]))
                }
                
                # Add position-weighted analysis if enabled
                if use_position_weighting and classifier:
                    # Simulate token analysis for database items
                    item_tokens = item_name.lower().split()
                    if len(item_tokens) > 0:
                        token_analysis = classifier.classify_hardware_tokens(item_tokens)
                        meta['hardware_relevance_score'] = token_analysis.get('hardware_relevance_score', 0.0)
                        meta['predicted_hardware'] = token_analysis.get('predicted_hardware', {})
                        meta['position_weighted'] = True
                    else:
                        meta['position_weighted'] = False
                else:
                    meta['position_weighted'] = False
                
                # Add additional columns if they exist
                for col in self.master_items.columns:
                    if col not in ['id', 'name', 'category', 'model', 'item']:
                        meta[col] = self.master_items.iloc[idx][col]
                
                metadata.append(meta)
                self.item_metadata[item_id] = meta
            
            logger.info(f"✅ Generated {embeddings.shape[0]} contextual embeddings of dimension {embeddings.shape[1]}")
            
            # Log some example contextual texts for verification
            logger.info("Sample contextual texts:")
            for i in range(min(3, len(contextual_texts))):
                logger.info(f"  {i+1}. {contextual_texts[i]}")
            
            return embeddings, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create FAISS index for fast similarity search.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        try:
            logger.info("📊 Creating FAISS index for fast similarity search...")
            
            dimension = embeddings.shape[1]
            n_items = embeddings.shape[0]
            
            # Choose index type based on dataset size
            if n_items < 1000:
                # Use flat index for small datasets (exact search)
                index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
                logger.info(f"Using IndexFlatIP for {n_items} items")
            else:
                # Use IVF index for larger datasets (approximate search)
                nlist = min(100, n_items // 10)  # Number of clusters
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                logger.info(f"Using IndexIVFFlat with {nlist} clusters for {n_items} items")
                
                # Train the index
                logger.info("Training FAISS index...")
                index.train(embeddings.astype('float32'))
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings.astype('float32'))
            
            # Add embeddings to index
            logger.info("Adding embeddings to index...")
            index.add(embeddings.astype('float32'))
            
            logger.info(f"✅ FAISS index created with {index.ntotal} vectors")
            
            return index
            
        except Exception as e:
            raise RuntimeError(f"Failed to create FAISS index: {e}")
    
    def save_database(self, 
                     embeddings: np.ndarray, 
                     metadata: List[Dict[str, Any]], 
                     faiss_index: faiss.Index,
                     use_position_weighting: bool = True) -> None:
        """
        Save the complete vector database to disk.
        
        Args:
            embeddings: Raw embeddings array
            metadata: Item metadata
            faiss_index: FAISS search index
        """
        try:
            logger.info("💾 Saving vector database...")
            
            # Save FAISS index
            faiss_path = self.database_output_path / "faiss_index.bin"
            faiss.write_index(faiss_index, str(faiss_path))
            logger.info(f"FAISS index saved to: {faiss_path}")
            
            # Save raw embeddings (for backup/analysis)
            embeddings_path = self.database_output_path / "embeddings.npy"
            np.save(embeddings_path, embeddings)
            logger.info(f"Raw embeddings saved to: {embeddings_path}")
            
            # Save metadata
            metadata_path = self.database_output_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved to: {metadata_path}")
            
            # Save item lookup table (for fast ID-based access)
            lookup_table = {meta['id']: meta for meta in metadata}
            lookup_path = self.database_output_path / "item_lookup.json"
            with open(lookup_path, 'w', encoding='utf-8') as f:
                json.dump(lookup_table, f, indent=2, ensure_ascii=False)
            logger.info(f"Item lookup table saved to: {lookup_path}")
            
            # Calculate category statistics for config
            category_stats = {}
            for item in metadata:
                category = item.get('category', 'Unknown')
                category_stats[category] = category_stats.get(category, 0) + 1
            
            # Save database configuration and statistics
            db_config = {
                'created_date': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'master_items_path': str(self.master_items_path),
                'total_items': len(metadata),
                'total_categories': len(category_stats),
                'category_distribution': category_stats,
                'embedding_dimension': embeddings.shape[1],
                'index_type': type(faiss_index).__name__,
                'database_version': '2.1',  # Updated version with position weighting
                'position_weighting_enabled': use_position_weighting
            }
            
            config_path = self.database_output_path / "database_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(db_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Database config saved to: {config_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save database: {e}")
    
    def test_database(self, faiss_index: faiss.Index, metadata: List[Dict[str, Any]]) -> None:
        """
        Test the created database with sample queries.
        
        Args:
            faiss_index: FAISS search index
            metadata: Item metadata
        """
        try:
            logger.info("🧪 Testing vector database...")
            
            # Test queries
            test_queries = [
                "Nintendo DS",
                "ニンテンドーDS", 
                "DS Lite",
                "Game Boy",
                "PlayStation",
                "controller",
                "コントローラー"
            ]
            
            for query in test_queries:
                logger.info(f"\n--- Query: '{query}' ---")
                
                # Encode query
                query_embedding = self.model.encode([query])
                faiss.normalize_L2(query_embedding.astype('float32'))
                
                # Search
                k = 3  # Top 3 results
                scores, indices = faiss_index.search(query_embedding.astype('float32'), k)
                
                # Display results
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx != -1:  # Valid result
                        item_meta = metadata[idx]
                        logger.info(f"  {i+1}. {item_meta['name']} (ID: {item_meta['id']}, Score: {score:.4f})")
            
        except Exception as e:
            logger.error(f"Database testing failed: {e}")
    
    def create_complete_database(self, use_position_weighting: bool = True) -> Dict[str, Any]:
        """
        Execute complete vector database creation pipeline.
        
        Returns:
            Database creation summary
        """
        try:
            logger.info("🚀 Starting Vector Database Creation Pipeline")
            
            # Step 1: Load model
            logger.info("\n📦 Step 1: Loading Trained Model")
            self.load_model()
            
            # Step 2: Load master items
            logger.info("\n📋 Step 2: Loading Master Items")
            self.load_master_items()
            
            # Step 3: Generate embeddings
            logger.info(f"\n🔄 Step 3: Generating Embeddings (Position Weighting: {use_position_weighting})")
            embeddings, metadata = self.generate_embeddings(use_position_weighting)
            
            # Step 4: Create FAISS index
            logger.info("\n📊 Step 4: Creating Search Index")
            faiss_index = self.create_faiss_index(embeddings)
            
            # Step 5: Save database
            logger.info("\n💾 Step 5: Saving Vector Database")
            self.save_database(embeddings, metadata, faiss_index, use_position_weighting)
            
            # Step 6: Test database
            logger.info("\n🧪 Step 6: Testing Database")
            self.test_database(faiss_index, metadata)
            
            # Create summary
            summary = {
                'creation_date': datetime.now().isoformat(),
                'total_items': len(metadata),
                'embedding_dimension': embeddings.shape[1],
                'database_path': str(self.database_output_path),
                'model_used': str(self.model_path),
                'master_items_file': str(self.master_items_path),
                'status': 'success'
            }
            
            logger.info("\n✅ Vector Database Creation Completed Successfully!")
            logger.info(f"📁 Database saved to: {self.database_output_path}")
            logger.info(f"📊 Total items indexed: {summary['total_items']}")
            logger.info(f"🔍 Ready for real-time classification!")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Database creation failed: {e}")
            raise


class VectorDatabaseLoader:
    """
    Utility class to load and use the created vector database.
    """
    
    def __init__(self, database_path: str, model_path: str):
        """
        Initialize database loader.
        
        Args:
            database_path: Path to vector database directory
            model_path: Path to trained model
        """
        self.database_path = Path(database_path)
        self.model_path = Path(model_path)
        
        self.model = None
        self.faiss_index = None
        self.metadata = None
        self.item_lookup = None
        
    def load_database(self) -> None:
        """Load the complete vector database."""
        try:
            logger.info(f"Loading vector database from: {self.database_path}")
            
            # Load model
            self.model = SentenceTransformer(str(self.model_path))
            
            # Load FAISS index
            faiss_path = self.database_path / "faiss_index.bin"
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            # Load metadata
            metadata_path = self.database_path / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Load lookup table
            lookup_path = self.database_path / "item_lookup.json"
            with open(lookup_path, 'r', encoding='utf-8') as f:
                self.item_lookup = json.load(f)
            
            # Build category statistics
            category_counts = {}
            for item in self.metadata:
                category = item.get('category', 'Unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            logger.info(f"Database loaded: {len(self.metadata)} items ready for search")
            logger.info(f"Categories available: {len(category_counts)}")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  {category}: {count} items")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load database: {e}")
    
    def search(self, query: str, k: int = 5, category_filter: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar items.
        
        Args:
            query: Search query text
            k: Number of results to return
            category_filter: Optional category to filter results by
            
        Returns:
            List of search results with category information
        """
        if self.faiss_index is None:
            raise RuntimeError("Database not loaded. Call load_database() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding.astype('float32'))
        
        # Search (get more results if filtering by category)
        search_k = k * 3 if category_filter else k
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                item_meta = self.metadata[idx].copy()
                item_meta['similarity_score'] = float(score)
                
                # Apply category filter if specified
                if category_filter:
                    if item_meta.get('category', '').lower() != category_filter.lower():
                        continue
                
                results.append(item_meta)
                
                # Stop if we have enough results
                if len(results) >= k:
                    break
        
        return results
    
    def get_categories(self) -> Dict[str, int]:
        """
        Get all available categories and their item counts.
        
        Returns:
            Dictionary of category names and counts
        """
        if self.metadata is None:
            raise RuntimeError("Database not loaded. Call load_database() first.")
        
        category_counts = {}
        for item in self.metadata:
            category = item.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
    
    def search_by_category(self, category: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Get all items from a specific category.
        
        Args:
            category: Category name to search for
            k: Maximum number of results to return
            
        Returns:
            List of items in the specified category
        """
        if self.metadata is None:
            raise RuntimeError("Database not loaded. Call load_database() first.")
        
        results = []
        for item in self.metadata:
            if item.get('category', '').lower() == category.lower():
                results.append(item.copy())
                if len(results) >= k:
                    break
        
        return results


def main():
    """Main function to create vector database."""
    
    # Configuration
    model_path = "intfloat/multilingual-e5-base"
    master_items_path = "/Users/jovitaeliana/Personal/strustore/items.json"  # Path to canonical taxonomy
    database_output_path = "models/vector_database"
    
    # Using pre-trained LaBSE model - no local file check needed
    
    # Check if master items file exists
    if not Path(master_items_path).exists():
        logger.error(f"Master items file not found at: {master_items_path}")
        logger.error("Please ensure the items.json file is available.")
        return
    
    # Create vector database
    creator = VectorDatabaseCreator(model_path, master_items_path, database_output_path)
    summary = creator.create_complete_database()
    
    return summary


if __name__ == '__main__':
    main()
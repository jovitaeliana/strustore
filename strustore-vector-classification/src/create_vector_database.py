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
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            logger.info(f"Loading trained model from: {self.model_path}")
            self.model = SentenceTransformer(str(self.model_path))
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def load_master_items(self) -> None:
        """Load master items from CSV file."""
        try:
            if not self.master_items_path.exists():
                raise FileNotFoundError(f"Master items file not found: {self.master_items_path}")
            
            logger.info(f"Loading master items from: {self.master_items_path}")
            self.master_items = pd.read_csv(self.master_items_path)
            
            # Validate required columns
            required_columns = ['id', 'item']
            missing_columns = [col for col in required_columns if col not in self.master_items.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and validate data
            self.master_items = self.master_items.dropna(subset=['item'])
            self.master_items['item'] = self.master_items['item'].astype(str).str.strip()
            
            # Remove empty items
            self.master_items = self.master_items[self.master_items['item'] != '']
            
            logger.info(f"Loaded {len(self.master_items)} items from master list")
            
            # Display sample items
            logger.info("Sample items:")\n            for idx, row in self.master_items.head().iterrows():
                logger.info(f"  ID {row['id']}: {row['item']}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load master items: {e}")
    
    def generate_embeddings(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:\n        """
        Generate embeddings for all master items.
        
        Returns:
            Tuple of (embeddings_array, metadata_list)
        """
        try:
            logger.info("🔄 Generating embeddings for master items...")
            
            items = self.master_items['item'].tolist()
            item_ids = self.master_items['id'].tolist()
            
            # Generate embeddings in batches for efficiency
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_items,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            
            # Create metadata for each item
            metadata = []
            for idx, (item_id, item_name) in enumerate(zip(item_ids, items)):
                meta = {
                    'id': item_id,
                    'name': item_name,
                    'vector_index': idx,
                    'embedding_norm': float(np.linalg.norm(embeddings[idx]))
                }
                
                # Add additional columns if they exist
                for col in self.master_items.columns:
                    if col not in ['id', 'item']:
                        meta[col] = self.master_items.iloc[idx][col]
                
                metadata.append(meta)
                self.item_metadata[item_id] = meta
            
            logger.info(f"✅ Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
            
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
                     faiss_index: faiss.Index) -> None:
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
            
            # Save database configuration and statistics
            db_config = {
                'created_date': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'master_items_path': str(self.master_items_path),
                'total_items': len(metadata),
                'embedding_dimension': embeddings.shape[1],
                'index_type': type(faiss_index).__name__,
                'database_version': '1.0'
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
    
    def create_complete_database(self) -> Dict[str, Any]:
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
            logger.info("\n🔄 Step 3: Generating Embeddings")
            embeddings, metadata = self.generate_embeddings()
            
            # Step 4: Create FAISS index
            logger.info("\n📊 Step 4: Creating Search Index")
            faiss_index = self.create_faiss_index(embeddings)
            
            # Step 5: Save database
            logger.info("\n💾 Step 5: Saving Vector Database")
            self.save_database(embeddings, metadata, faiss_index)
            
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
            
            logger.info(f"Database loaded: {len(self.metadata)} items ready for search")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load database: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar items.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if self.faiss_index is None:
            raise RuntimeError("Database not loaded. Call load_database() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding.astype('float32'))
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                item_meta = self.metadata[idx].copy()
                item_meta['similarity_score'] = float(score)
                results.append(item_meta)
        
        return results


def main():
    """Main function to create vector database."""
    
    # Configuration
    model_path = "models/gaming-console-semantic-model"
    master_items_path = "../../yjpa_scraper/items.csv"  # Path to your items.csv
    database_output_path = "models/vector_database"
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Trained model not found at: {model_path}")
        logger.error("Please run train_model.py first to create the model.")
        return
    
    # Check if master items file exists
    if not Path(master_items_path).exists():
        logger.error(f"Master items file not found at: {master_items_path}")
        logger.error("Please ensure the items.csv file is available.")
        return
    
    # Create vector database
    creator = VectorDatabaseCreator(model_path, master_items_path, database_output_path)
    summary = creator.create_complete_database()
    
    return summary


if __name__ == '__main__':
    main()
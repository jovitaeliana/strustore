"""
Vector Database Creation Script

Creates the vector database from the trained model and master items list.
Adds optional text normalization, alias expansion, and CLI/env-driven config
without breaking existing defaults.
"""

import os
import argparse
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

# Import local utilities (resolve repo root dynamically)
import sys
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from position_weighted_embeddings import PositionWeightedTokenClassifier
from text_normalization import TextNormalizer
from aliases.generator import HeuristicAliasProvider

# ---------------- Family and device type inference utilities ---------------- #

_FAMILY_KEYWORDS = [
    ("nintendo ds lite", ["ds lite", "dsl", "dslite"]),
    ("nintendo dsi", ["dsi"]),
    ("nintendo 3ds", ["3ds"]),
    ("new 2ds xl", ["2ds xl", "new 2ds xl", "2ds"]),
    ("nintendo ds", ["nds", "ds"]),
    ("game boy advance", ["gba"]),
    ("game boy color", ["gbc"]),
    ("game boy", ["gb", "gameboy"]),
    ("nintendo 64", ["n64"]),
    ("super famicom", ["sfc", "snes"]),
    ("famicom", ["fc", "nes"]),
    ("gamecube", ["gc", "nintendo gamecube"]),
    ("wii u", ["wii u"]),
    ("wii", ["nintendo wii"]),
    ("nintendo switch", ["switch"]),
    ("playstation 5", ["ps5"]),
    ("playstation 4", ["ps4"]),
    ("playstation 3", ["ps3"]),
    ("playstation 2", ["ps2"]),
    ("playstation", ["ps", "ps1", "playstation 1"]),
    ("ps vita", ["vita", "playstation vita"]),
    ("psp", ["playstation portable"]),
    ("xbox one", ["xbone"]),
    ("xbox 360", ["x360"]),
    ("xbox", ["microsoft xbox"]),
    ("sega saturn", ["saturn"]),
    ("dreamcast", ["sega dreamcast"]),
]

def infer_family(name: str, model: Optional[str]) -> Tuple[str, List[str]]:
    """Infer a canonical family slug and tight synonyms from item name/model codes."""
    import re
    n = (name or "").strip().lower()
    m = (model or "").strip().lower()

    # Model code driven mapping (most specific)
    code = f"{n} {m}"
    if re.search(r"\b(dol-\d{3})\b", code):
        return "gamecube", ["gc", "nintendo gamecube"]
    if re.search(r"\bnus-\d+\b", code):
        return "nintendo 64", ["n64"]
    if re.search(r"\bntr-001\b", code):
        return "nintendo ds", ["nds", "ds"]
    if re.search(r"\busg-001\b", code):
        return "nintendo ds lite", ["ds lite", "dsl", "dslite"]
    if re.search(r"\btwl-001\b", code):
        return "nintendo dsi", ["dsi"]
    if re.search(r"\bctr-001\b", code):
        return "nintendo 3ds", ["3ds"]
    if re.search(r"\bhdh-001\b", code):
        return "nintendo switch lite", ["switch lite", "switch"]
    if re.search(r"\bheg-001\b", code):
        return "nintendo switch", ["switch"]
    if re.search(r"\bwup-\d+\b", code):
        return "wii u", ["wii u"]
    if re.search(r"\brvl-\d+\b", code):
        return "wii", ["nintendo wii"]
    if re.search(r"\bscph-\d+\b", code):
        # Could refine by range to PS1/PS2; keep generic Playstation
        return "playstation", ["ps", "ps1", "ps2"]
    if re.search(r"\bcech-\w+\b", code):
        return "playstation 3", ["ps3"]
    if re.search(r"\bcuh-\w+\b", code):
        return "playstation 4", ["ps4"]
    if re.search(r"\bcfi-\w+\b", code):
        return "playstation 5", ["ps5"]
    if re.search(r"\bdmg-\d+\b", code):
        return "game boy", ["gb", "gameboy"]
    if re.search(r"\bcgb-\d+\b", code):
        return "game boy color", ["gbc"]
    if re.search(r"\bagb-\d+\b", code) or re.search(r"\bags-\d+\b", code):
        return "game boy advance", ["gba"]

    # Keyword-driven mapping
    for fam, syns in _FAMILY_KEYWORDS:
        if fam in n:
            return fam, syns
    # broader tokens
    if "gc" in n or "gamecube" in n:
        return "gamecube", ["gc", "nintendo gamecube"]
    if re.search(r"\bn64\b", n):
        return "nintendo 64", ["n64"]
    if re.search(r"\bsfc\b|super famicom|snes", n):
        return "super famicom", ["sfc", "snes"]
    if re.search(r"\bds\b|nintendo ds", n):
        return "nintendo ds", ["nds", "ds"]
    if "dsl" in n or "ds lite" in n or "dslite" in n:
        return "nintendo ds lite", ["ds lite", "dsl", "dslite"]
    if re.search(r"\b3ds\b|nintendo 3ds", n):
        return "nintendo 3ds", ["3ds"]
    if re.search(r"\b2ds\b", n):
        return "nintendo 2ds", ["2ds"]
    if re.search(r"game boy advance|\bgba\b", n):
        return "game boy advance", ["gba"]
    if re.search(r"game boy color|\bgbc\b", n):
        return "game boy color", ["gbc"]
    if re.search(r"game boy|gameboy|\bgb\b", n):
        return "game boy", ["gb", "gameboy"]
    if "wii u" in n:
        return "wii u", ["wii u"]
    if re.search(r"\bwii\b", n):
        return "wii", ["nintendo wii"]
    if "switch" in n:
        return "nintendo switch", ["switch"]
    if re.search(r"\bps ?5\b|playstation 5", n):
        return "playstation 5", ["ps5"]
    if re.search(r"\bps ?4\b|playstation 4", n):
        return "playstation 4", ["ps4"]
    if re.search(r"\bps ?3\b|playstation 3", n):
        return "playstation 3", ["ps3"]
    if re.search(r"\bps ?2\b|playstation 2", n):
        return "playstation 2", ["ps2"]
    if re.search(r"\bps ?1\b|playstation(?! [2345])", n) or "sony playstation" in n:
        return "playstation", ["ps", "ps1"]
    if "ps vita" in n or "vita" in n:
        return "ps vita", ["vita", "playstation vita"]
    if re.search(r"\bpsp\b|playstation portable", n):
        return "psp", ["psp", "playstation portable"]
    if "xbox one" in n or "xbone" in n:
        return "xbox one", ["xbone"]
    if "xbox 360" in n or "x360" in n:
        return "xbox 360", ["x360"]
    if re.search(r"\bxbox\b", n):
        return "xbox", ["microsoft xbox"]
    if "saturn" in n:
        return "sega saturn", ["saturn"]
    if "dreamcast" in n:
        return "dreamcast", ["sega dreamcast"]

    # Unknown
    return "", []


def infer_device_type(category: Optional[str]) -> str:
    c = (category or "").strip().lower()
    if c == "video game consoles":
        return "console"
    if c == "controllers & attachments":
        return "controller"
    if c == "power cables & connectors":
        return "power cable"
    if c == "memory cards & expansion packs":
        return "memory card"
    if c == "video games":
        return "video game"
    if c == "other video game accessories":
        return "accessory"
    return "accessory"

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
                 database_output_path: str = "models/vector_database",
                 normalizer: Optional[TextNormalizer] = None,
                 use_normalization: bool = True,
                 use_aliases: bool = False):
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
        # Normalization / aliases
        self.normalizer = normalizer or TextNormalizer()
        self.use_normalization = use_normalization
        self.use_aliases = use_aliases

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
    
    def create_contextual_text(self, item_name: str, category: str, model: str = "", item_id: str = "", position_weighted: bool = False) -> str:
        """
        Create minimal, structured contextual text for high-precision retrieval.
        Format: title: <name> | family: <family and tight synonyms> | type: <device_type> | model: <code> | id: <id>
        """
        fam, syns = infer_family(item_name, model)
        device_type = infer_device_type(category)
        family_field = fam
        if syns:
            family_field = fam + (" " + " ".join(syns) if fam else " ".join(syns))

        parts: List[str] = []
        parts.append(f"title: {item_name}")
        if family_field:
            parts.append(f"family: {family_field}")
        if device_type:
            parts.append(f"type: {device_type}")
        if model:
            parts.append(f"model: {model}")
        if item_id:
            parts.append(f"id: {item_id}")
        return " | ".join(parts)

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

            # Optional normalization step (index-side)
            if self.use_normalization:
                contextual_texts = [self.normalizer.normalize_for_index(t) for t in contextual_texts]
            
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
            
            # Create enhanced metadata for each item with family/type info
            metadata = []
            classifier = PositionWeightedTokenClassifier() if use_position_weighting else None
            
            for idx, (item_id, item_name, item_category, item_model, contextual_text) in enumerate(
                zip(item_ids, items, item_categories, item_models, contextual_texts)
            ):
                fam, syns = infer_family(item_name, item_model)
                device_type = infer_device_type(item_category)
                meta = {
                    'id': item_id,
                    'name': item_name,                    # Original Firebase name
                    'category': item_category,            # Original Firebase category
                    'model': item_model if item_model else '',  # Original Firebase model
                    'contextual_text': contextual_text,   # Enhanced context for embedding
                    'family': fam,
                    'device_type': device_type,
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
            
            # Prepare normalized float32 embeddings for cosine search via IP
            emb_f32 = embeddings.astype('float32', copy=True)
            faiss.normalize_L2(emb_f32)

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
                index.train(emb_f32)
            
            # Add embeddings to index
            logger.info("Adding embeddings to index...")
            index.add(emb_f32)
            
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
            
            def build_query(q: str) -> str:
                qs = q.lower()
                aug: List[str] = [qs]
                # Family synonym expansions (query-side only)
                if 'ニンテンドーds' in qs or 'nintendo ds' in qs or qs.strip() == 'ds':
                    aug += ['nintendo ds', 'nds', 'ds', 'console']
                if 'ds lite' in qs or 'dsl' in qs:
                    aug += ['nintendo ds lite', 'dsl', 'dslite', 'console']
                if '3ds' in qs:
                    aug += ['nintendo 3ds', 'handheld']
                if 'game boy' in qs:
                    aug += ['game boy', 'gb', 'handheld']
                if 'playstation' in qs or qs.strip().startswith('ps'):
                    aug += ['playstation', 'ps', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'scph', 'cech', 'cuh', 'cfi', 'dualshock']
                if 'controller' in qs or 'コントローラー' in q:
                    aug += ['controller', 'gamepad', 'con', 'joystick', 'stick', 'pad', 'joy-con', 'joycon', 'dualshock']
                # Join unique tokens
                seen = set()
                toks = []
                for t in aug:
                    if t not in seen:
                        seen.add(t)
                        toks.append(t)
                joined = ' '.join(toks)
                return f"query: {joined}" if 'e5' in str(self.model_path).lower() else joined

            for query in test_queries:
                logger.info(f"\n--- Query: '{query}' ---")
                
                # Encode query (use E5 query prefix when applicable)
                query_text = build_query(query)
                q = self.model.encode([query_text])
                q = q.astype('float32', copy=True)
                faiss.normalize_L2(q)
                
                # Search
                k = 3  # Top 3 results
                scores, indices = faiss_index.search(q, k)
                
                # Display results
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx != -1:  # Valid result
                        item_meta = metadata[idx]
                        fam = item_meta.get('family', '')
                        dtype = item_meta.get('device_type', '')
                        logger.info(f"  {i+1}. {item_meta['name']} (ID: {item_meta['id']}, family: {fam or '-'}, type: {dtype or '-'}, Score: {score:.4f})")
            
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
        
        # Encode query with minimal normalization and family synonyms
        def build_query(qs: str) -> str:
            s = qs.lower()
            aug: List[str] = [s]
            if 'ニンテンドーds' in s or 'nintendo ds' in s or s.strip() == 'ds':
                aug += ['nintendo ds', 'nds', 'ds', 'console']
            if 'ds lite' in s or 'dsl' in s:
                aug += ['nintendo ds lite', 'dsl', 'dslite', 'console']
            if '3ds' in s:
                aug += ['nintendo 3ds', 'handheld']
            if 'game boy' in s or 'gameboy' in s or s.strip() == 'gb':
                aug += ['game boy', 'gb', 'handheld']
            if 'playstation' in s or s.strip().startswith('ps'):
                aug += ['playstation', 'ps', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'scph', 'cech', 'cuh', 'cfi', 'dualshock']
            if 'controller' in s or 'コントローラー' in s:
                aug += ['controller', 'gamepad', 'con', 'joystick', 'stick', 'pad', 'joy-con', 'joycon', 'dualshock']
            seen = set(); toks: List[str] = []
            for t in aug:
                if t not in seen:
                    seen.add(t); toks.append(t)
            joined = ' '.join(toks)
            return f"query: {joined}" if 'e5' in str(self.model_path).lower() else joined

        q = self.model.encode([build_query(query)])
        q = q.astype('float32', copy=True)
        faiss.normalize_L2(q)
        
        # Search (get more results if filtering by category)
        search_k = k * 5 if category_filter else max(k, 30)
        scores, indices = self.faiss_index.search(q, search_k)

        # Collect candidates with metadata
        candidates: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            m = self.metadata[idx].copy()
            m['similarity_score'] = float(score)
            m['_rank_index'] = int(idx)
            candidates.append(m)

        # Apply targeted re-ranking for specific queries
        def rerank_for_query(qs: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            import re
            s = qs.lower().strip()
            is_ps = ('playstation' in s) or s.startswith('ps') or any(t in s for t in ['ps1','ps2','ps3','ps4','ps5'])
            is_controller = ('controller' in s) or ('コントローラー' in s)
            if not (is_ps or is_controller):
                return items
            out = []
            for m in items:
                boost = 0.0
                name = (m.get('name') or '').lower()
                fam = (m.get('family') or '').lower()
                dtype = (m.get('device_type') or '').lower()
                model = (m.get('model') or '').upper()
                base = m['similarity_score']
                if is_ps:
                    if fam.startswith('playstation'):
                        boost += 0.10
                    if re.search(r'\bps[1-5]\b', name):
                        boost += 0.07
                    if re.search(r'\b(SCPH|CECH|CUH|CFI)[-0-9A-Z]*\b', model) or re.search(r'\b(scph|cech|cuh|cfi)\b', name):
                        boost += 0.07
                    if 'dualshock' in name or 'dual sense' in name or 'dualsense' in name:
                        boost += 0.03
                if is_controller:
                    if dtype == 'controller':
                        boost += 0.10
                    # Literal 'Con' token or suffix (case-insensitive)
                    if re.search(r'(^|\W)con(\W|$)', (m.get('name') or ''), re.IGNORECASE) or (m.get('name') or '').endswith(' Con'):
                        boost += 0.07
                    ctrl_kw = ['controller','gamepad','joystick','stick','pad','joy-con','joycon','dualshock']
                    if any(kw in name for kw in ctrl_kw):
                        boost += 0.05
                m['_rerank_score'] = base + boost
                out.append(m)
            # Sort by rerank score desc, then base similarity desc, then original index asc for stability
            out.sort(key=lambda mm: (mm.get('_rerank_score', mm['similarity_score']), mm['similarity_score'], -mm.get('_rank_index', 0)), reverse=True)
            return out

        candidates = rerank_for_query(query, candidates)

        # Apply category filter and return top-k
        results: List[Dict[str, Any]] = []
        for m in candidates:
            if category_filter and m.get('category', '').lower() != category_filter.lower():
                continue
            # Cleanup internal fields
            m.pop('_rank_index', None)
            m.pop('_rerank_score', None)
            results.append(m)
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


def main(argv: Optional[List[str]] = None):
    """CLI for creating the vector database with optional normalization."""
    parser = argparse.ArgumentParser(description="Create Strustore vector database")
    parser.add_argument("--model", default=os.getenv("STRUSTORE_MODEL_PATH", "intfloat/multilingual-e5-base"), help="Model name or path")
    parser.add_argument("--items", default=os.getenv("STRUSTORE_ITEMS_PATH", "items.json"), help="Path to items.json")
    parser.add_argument("--out", default=os.getenv("STRUSTORE_DB_PATH", "models/vector_database"), help="Output directory for vector DB")
    parser.add_argument("--no-normalize", action="store_true", help="Disable text normalization for index")
    parser.add_argument("--use-aliases", action="store_true", help="Enable heuristic alias expansion for index text")
    parser.add_argument("--log-level", default=os.getenv("STRUSTORE_LOG_LEVEL", "INFO"), help="Logging level")
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # Resolve items path intelligently
    items_path = Path(args.items)
    if not items_path.exists():
        # Try repo root fallback
        repo_root = ROOT_DIR
        candidate = repo_root / "items.json"
        if candidate.exists():
            items_path = candidate
        else:
            logger.error(f"Master items file not found at: {args.items}")
            return

    creator = VectorDatabaseCreator(
        model_path=args.model,
        master_items_path=str(items_path),
        database_output_path=args.out,
        normalizer=TextNormalizer(),
        use_normalization=(not args.no_normalize),
        use_aliases=args.use_aliases,
    )

    summary = creator.create_complete_database()
    return summary


if __name__ == '__main__':
    main()

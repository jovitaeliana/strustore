"""
GroundingDINO Result Enhancement Pipeline

Enhances GDINO detections using the vector database for semantic classification.
Adds optional token normalization and produces an `enhanced_classification` map per detection.
Itemtypes are used for display-only names; classification IDs come from items.json.
"""

import os
import re
import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

# Resolve repo root for local imports
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from text_normalization import TextNormalizer

# Dynamic weighting system is implemented within this class

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GdinoResultEnhancer:
    """
    Enhances GroundingDINO detection results using vector database semantic search.
    """
    
    def __init__(self, 
                 vector_db_path: str = "../../models/vector_database",
                 model_path: str = "intfloat/multilingual-e5-base",
                 gdino_output_dir: str = "../../gdinoOutput/final",
                 itemtypes_path: str = "../../itemtypes.json",
                 normalize_tokens: bool = True,
                 itemtypes_display_only: bool = True,
                 write_enhanced_classification: bool = True,
                 official_threshold: float = 0.60,
                 enforce_official_consistency: bool = True):
        """
        Initialize the result enhancer.
        
        Args:
            vector_db_path: Path to vector database directory
            model_path: Path to trained semantic model
            gdino_output_dir: Path to GroundingDINO output directory
            itemtypes_path: Path to itemtypes.json file
        """
        self.vector_db_path = Path(vector_db_path)
        self.model_path = model_path
        self.gdino_output_dir = Path(gdino_output_dir)
        self.itemtypes_path = Path(itemtypes_path)
        
        # Initialize components
        self.model = None
        self.faiss_index = None
        self.metadata = None
        self.item_lookup = None
        self.itemtypes_data = None
        self.itemtypes_embeddings = None
        self.itemtypes_index = None
        
        # Normalization and policy flags
        self.normalizer = TextNormalizer()
        self.normalize_tokens = normalize_tokens
        self.itemtypes_display_only = itemtypes_display_only
        self.write_enhanced_classification = write_enhanced_classification
        self.official_threshold = official_threshold
        self.enforce_official_consistency = enforce_official_consistency
        
        # Dynamic weighting system is implemented within class methods
        
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
    
    def load_itemtypes_database(self) -> None:
        """Load and process itemtypes.json for enhanced classification."""
        try:
            if not self.itemtypes_path.exists():
                logger.warning(f"Itemtypes file not found: {self.itemtypes_path}")
                return
            
            logger.info(f"🔄 Loading itemtypes database from: {self.itemtypes_path}")
            
            # Load itemtypes data
            with open(self.itemtypes_path, 'r', encoding='utf-8') as f:
                self.itemtypes_data = json.load(f)
            
            # Flatten all items into a single list for embedding
            all_itemtypes = []
            for category_items in self.itemtypes_data.values():
                all_itemtypes.extend(category_items)
            
            logger.info(f"Found {len(all_itemtypes)} itemtypes across {len(self.itemtypes_data)} categories")
            
            # Create enhanced contextual texts for itemtypes
            itemtype_texts = []
            itemtype_metadata = []
            
            for item in all_itemtypes:
                # Create rich contextual text
                context_parts = []
                context_parts.append(item['name'])
                
                # Add all official names
                if item.get('official_names'):
                    context_parts.extend(item['official_names'])
                
                # Add brand and console info
                if item.get('brand'):
                    context_parts.append(f"brand: {item['brand']}")
                if item.get('console'):
                    context_parts.append(f"console: {item['console']}")
                
                # Add model codes
                if item.get('model_codes'):
                    context_parts.extend([f"model: {code}" for code in item['model_codes']])
                
                # Add keywords
                if item.get('keywords'):
                    context_parts.extend(item['keywords'])
                
                # Add category context
                context_parts.append(f"category: {item['category']}")
                
                contextual_text = ' | '.join(context_parts)
                itemtype_texts.append(contextual_text)
                
                # Store metadata
                meta = {
                    'id': item['id'],
                    'name': item['name'],
                    'category': item['category'],
                    'brand': item.get('brand', ''),
                    'console': item.get('console', ''),
                    'contextual_text': contextual_text,
                    'source': 'itemtypes'
                }
                itemtype_metadata.append(meta)
            
            # Generate embeddings for itemtypes
            logger.info("Generating embeddings for itemtypes...")
            
            # Add E5 prefix for better retrieval performance
            prefixed_texts = [f"passage: {text}" for text in itemtype_texts]
            
            # Generate embeddings in batches
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(prefixed_texts), batch_size):
                batch_texts = prefixed_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
            
            self.itemtypes_embeddings = np.vstack(all_embeddings)
            
            # Create FAISS index for itemtypes
            dimension = self.itemtypes_embeddings.shape[1]
            self.itemtypes_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity (in-place on a float32 copy)
            emb_f32 = self.itemtypes_embeddings.astype('float32', copy=True)
            faiss.normalize_L2(emb_f32)
            self.itemtypes_index.add(emb_f32)
            
            # Store metadata
            self.itemtypes_metadata = itemtype_metadata
            
            logger.info(f"✅ Itemtypes database loaded: {len(itemtype_metadata)} items with {dimension}D embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load itemtypes database: {e}")
            # Continue without itemtypes support
    
    def apply_dynamic_token_weighting(self, tokens: List[str], max_tokens: int = 10, 
                                     decay_factor: float = 0.1, min_weight: float = 0.1) -> List[Dict[str, Any]]:
        """
        Apply dynamic priority-based token weighting using exponential decay.
        
        Args:
            tokens: Ordered list of tokens (index 0 = highest priority)
            max_tokens: Maximum number of tokens to consider
            decay_factor: Exponential decay rate for token weights
            min_weight: Minimum weight threshold for inclusion
            
        Returns:
            List of weighted tokens with metadata
        """
        if not tokens:
            return []
        
        # Noise terms that should be filtered out regardless of position
        noise_terms = {
            'de', 'en', 'com', 'por', 'para', 'con', 'las', 'los', 'el', 'la', 'w', 'r', 'l', 
            'buy', 'best', 'online', 'youtube', 'tiktok', 'facebook', 'mercadolibre', 'wallapop', 
            'amazon', 'ebay', 'price', 'precio', 'estado', 'condition', 'new', 'used', 'segunda', 
            'mano', 'original', 'genuine', 'oem', 'tested', 'working', 'bundle', 'set', 'kit', 
            'box', 'caja', 'メルカリ', 'ラクマ', 'yahoo', 'オークション', 'フリマ', 'eur'
        }
        
        # Industry-specific boost keywords
        hardware_keywords = {
            'nintendo', 'playstation', 'xbox', 'switch', 'controller', 'console',
            'ds', 'dsi', 'psp', 'vita', 'gamecube', 'wii', 'joy', 'con', 'joycon',
            'ps1', 'ps2', 'ps3', 'ps4', 'ps5', '3ds', 'gba', 'n64', 'snes',
            'scph', 'oled', 'pro', 'slim', 'lite', 'memory', 'card', 'handheld',
            'dualshock', 'sixaxis', 'dualsense', 'gameboy', 'saturn', 'genesis'
        }
        
        weighted_tokens = []
        
        for idx, token in enumerate(tokens[:max_tokens]):
            token_clean = token.strip().lower()
            
            # Skip empty or very short tokens
            if len(token_clean) < 2:
                continue
                
            # Skip noise terms
            if token_clean in noise_terms:
                continue
            
            # Calculate exponential decay weight based on position
            # Weight = 1.0 * e^(-decay_factor * index)
            position_weight = np.exp(-decay_factor * idx)
            
            # Apply hardware boost if token is hardware-related
            hardware_boost = 1.5 if any(hw in token_clean for hw in hardware_keywords) else 1.0
            
            # Calculate final weight
            final_weight = position_weight * hardware_boost
            
            # Only include tokens above minimum weight threshold
            if final_weight >= min_weight:
                weighted_tokens.append({
                    'token': token,
                    'position_index': idx,
                    'position_weight': position_weight,
                    'hardware_boost': hardware_boost,
                    'final_weight': final_weight,
                    'is_hardware': hardware_boost > 1.0,
                    'token_clean': token_clean
                })
        
        # Sort by final weight (descending)
        weighted_tokens.sort(key=lambda x: x['final_weight'], reverse=True)
        
        return weighted_tokens

    def search_similar_items(self, tokens: List[str], k: int = 5, use_dynamic_weighting: bool = True, 
                           weighting_config: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar items based on token array using dynamic priority-based weighting.
        
        Args:
            tokens: Ordered list of detection tokens (index 0 = highest priority)
            k: Number of similar items to return
            use_dynamic_weighting: Enable dynamic token weighting system
            weighting_config: Configuration for weighting algorithm
            
        Returns:
            List of similar items with similarity scores
        """
        if not tokens:
            return []
        
        # Default weighting configuration
        if weighting_config is None:
            weighting_config = {
                'max_tokens': 10,
                'decay_factor': 0.1,
                'min_weight': 0.1,
                'hardware_boost': 1.5
            }
        
        try:
            # Optional normalization applied before weighting
            if self.normalize_tokens:
                tokens = self.normalizer.normalize_tokens(tokens)

            if use_dynamic_weighting:
                # Apply dynamic priority-based weighting
                weighted_tokens = self.apply_dynamic_token_weighting(
                    tokens, 
                    max_tokens=weighting_config['max_tokens'],
                    decay_factor=weighting_config['decay_factor'],
                    min_weight=weighting_config['min_weight']
                )
                
                if not weighted_tokens:
                    logger.debug("No valid tokens after dynamic weighting")
                    return []
                
                # Create query from weighted tokens (prioritize high-weight tokens)
                query_tokens = [wt['token'] for wt in weighted_tokens[:15]]
                
                logger.debug(f"Dynamic weighting applied: {len(weighted_tokens)} valid tokens")
                # Fixed logging statement
                top_tokens_str = ', '.join([f"{wt['token']}({wt['final_weight']:.2f})" for wt in weighted_tokens[:5]])
                logger.debug(f"Top 5 weighted tokens: {top_tokens_str}")
                
            else:
                # Fallback to simple filtering
                filtered_tokens = []
                for token in tokens[:15]:
                    token_clean = token.strip().lower()
                    if len(token_clean) >= 2:
                        filtered_tokens.append(token)
                
                query_tokens = filtered_tokens
            
            if not query_tokens:
                return []
            
            # Create search query (normalize once more)
            if self.normalize_tokens:
                query_tokens = self.normalizer.normalize_tokens(query_tokens)
            query_text = ' '.join(query_tokens)
            
            # Add E5 query prefix for better retrieval performance
            query_text = f"query: {query_text}"
            
            # Encode query
            q = self.model.encode([query_text])
            q = q.astype('float32', copy=True)
            faiss.normalize_L2(q)
            
            # Search
            scores, indices = self.faiss_index.search(q, k)
            
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

    def search_itemtypes(self, tokens: List[str], k: int = 5, use_dynamic_weighting: bool = True,
                       weighting_config: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar items in the itemtypes database using dynamic priority-based weighting.
        
        Args:
            tokens: Ordered list of detection tokens (index 0 = highest priority)
            k: Number of similar items to return
            use_dynamic_weighting: Enable dynamic token weighting system
            weighting_config: Configuration for weighting algorithm
            
        Returns:
            List of similar items with similarity scores
        """
        if not tokens or self.itemtypes_index is None:
            return []
        
        # Default weighting configuration
        if weighting_config is None:
            weighting_config = {
                'max_tokens': 10,
                'decay_factor': 0.1,
                'min_weight': 0.1,
                'hardware_boost': 1.5
            }
        
        try:
            if use_dynamic_weighting:
                # Apply dynamic priority-based weighting
                weighted_tokens = self.apply_dynamic_token_weighting(
                    tokens, 
                    max_tokens=weighting_config['max_tokens'],
                    decay_factor=weighting_config['decay_factor'],
                    min_weight=weighting_config['min_weight']
                )
                
                if not weighted_tokens:
                    logger.debug("Itemtypes: No valid tokens after dynamic weighting")
                    return []
                
                # Create query from weighted tokens
                query_tokens = [wt['token'] for wt in weighted_tokens[:15]]
                
                logger.debug(f"Itemtypes dynamic weighting: {len(weighted_tokens)} valid tokens")
                
            else:
                # Fallback to simple filtering
                filtered_tokens = []
                for token in tokens[:15]:
                    token_clean = token.strip().lower()
                    if len(token_clean) >= 2:
                        filtered_tokens.append(token)
                
                query_tokens = filtered_tokens
            
            if not query_tokens:
                return []
            
            # Create search query
            query_text = ' '.join(query_tokens)
            
            # Add E5 query prefix for better retrieval performance
            query_text = f"query: {query_text}"
            
            # Encode query
            q = self.model.encode([query_text])
            q = q.astype('float32', copy=True)
            faiss.normalize_L2(q)
            
            # Search
            scores, indices = self.itemtypes_index.search(q, k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score > 0.0:  # Valid result with positive similarity
                    item_meta = self.itemtypes_metadata[idx].copy()
                    item_meta['similarity_score'] = float(score)
                    results.append(item_meta)
            
            return results
            
        except Exception as e:
            logger.error(f"Itemtypes search failed: {e}")
            return []

    def find_itemtypes_match(self, tokens: List[str], min_similarity: float = 0.3, use_dynamic_weighting: bool = True) -> Optional[Dict[str, Any]]:
        """
        Find the best itemtypes match for given tokens using dynamic weighting.
        
        Args:
            tokens: Detection tokens
            min_similarity: Minimum similarity threshold
            use_dynamic_weighting: Enable dynamic token weighting system
            
        Returns:
            Best itemtypes match or None
        """
        if not self.itemtypes_index:
            return None
            
        itemtype_results = self.search_itemtypes(tokens, k=3, use_dynamic_weighting=use_dynamic_weighting)
        if itemtype_results and itemtype_results[0]['similarity_score'] >= min_similarity:
            # Add dynamic weighting analysis metadata if enabled
            if use_dynamic_weighting:
                # Add metadata about token prioritization
                weighted_tokens = self.apply_dynamic_token_weighting(tokens)
                if weighted_tokens:
                    itemtype_results[0]['dynamic_weighting_analysis'] = {
                        'top_weighted_tokens': [wt['token'] for wt in weighted_tokens[:5]],
                        'total_weighted_tokens': len(weighted_tokens),
                        'highest_weight': weighted_tokens[0]['final_weight'] if weighted_tokens else 0.0,
                        'has_hardware_terms': any(wt['is_hardware'] for wt in weighted_tokens)
                    }
            return itemtype_results[0]
        return None

    def assess_token_quality(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Assess the quality of tokens for classification purposes.
        
        Args:
            tokens: List of tokens to assess
            
        Returns:
            Dictionary with quality metrics and filtered tokens
        """
        if not tokens:
            return {'quality_score': 0.0, 'valid_tokens': [], 'has_hardware_terms': False}
        
        # Gaming-specific keywords for relevance check
        gaming_keywords = {
            'nintendo', 'playstation', 'xbox', 'switch', 'controller', 'console',
            'ds', 'dsi', 'psp', 'vita', 'gamecube', 'wii', 'joy', 'con', 'joycon',
            'ps1', 'ps2', 'ps3', 'ps4', 'ps5', '3ds', 'gba', 'n64', 'snes',
            'scph', 'oled', 'pro', 'slim', 'lite', 'memory', 'card', 'handheld',
            'dualshock', 'sixaxis', 'dualsense', 'gameboy', 'saturn', 'genesis'
        }
        
        # Invalid token patterns (garbled text, special characters, etc.)
        invalid_patterns = [
            r'^[■▪▫□◯○●◆◇△▲▼◄►♠♥♦♣]+$',  # Special characters only
            r'^[\d\s\W]+$',  # Only numbers/spaces/special chars
            r'^.{1}$',  # Single character tokens
            r'^[xX]+$',  # Just x's
        ]
        
        # Common irrelevant terms that should lower quality
        irrelevant_terms = {
            'yahoo', 'ebay', 'amazon', 'mercado', 'libre', 'wallapop', 'facebook',
            'tiktok', 'youtube', 'instagram', 'twitter', 'com', 'www', 'http',
            'price', 'precio', 'estado', 'condition', 'shipping', 'free', 'best',
            'buy', 'sale', 'nuevo', 'usado', 'second', 'hand', 'mano', 'de',
            'en', 'el', 'la', 'los', 'las', 'por', 'para', 'con', 'sin',
            '新品', '中古', '美品', 'オークション', 'メルカリ', 'ラクマ'
        }
        
        valid_tokens = []
        hardware_tokens = []
        quality_score = 0.0
        
        import re
        
        for token in tokens:
            token_clean = token.strip().lower()
            
            # Skip empty or very short tokens
            if len(token_clean) < 2:
                continue
            
            # Check for invalid patterns
            is_invalid = any(re.match(pattern, token, re.IGNORECASE) for pattern in invalid_patterns)
            if is_invalid:
                quality_score -= 0.5  # Penalize invalid tokens
                continue
            
            # Check if token is irrelevant
            if token_clean in irrelevant_terms:
                quality_score -= 0.2  # Light penalty for irrelevant terms
                continue
            
            # Valid token - add to list
            valid_tokens.append(token)
            
            # Check if it's a hardware-related token
            if any(keyword in token_clean for keyword in gaming_keywords):
                hardware_tokens.append(token)
                quality_score += 1.0  # Boost for hardware relevance
            else:
                quality_score += 0.3  # Small boost for general valid tokens
        
        # Calculate final quality metrics
        hardware_ratio = len(hardware_tokens) / len(tokens) if tokens else 0
        valid_ratio = len(valid_tokens) / len(tokens) if tokens else 0
        
        # Final quality score (0-1 scale)
        final_quality = min(1.0, max(0.0, (quality_score + hardware_ratio * 2 + valid_ratio) / 4))
        
        return {
            'quality_score': final_quality,
            'valid_tokens': valid_tokens,
            'hardware_tokens': hardware_tokens,
            'has_hardware_terms': len(hardware_tokens) > 0,
            'hardware_ratio': hardware_ratio,
            'valid_ratio': valid_ratio,
            'total_tokens': len(tokens),
            'valid_token_count': len(valid_tokens),
            'hardware_token_count': len(hardware_tokens)
        }

    def get_best_classification(self, tokens: List[str], min_similarity: float = 0.55, 
                              use_dynamic_weighting: bool = True, 
                              weighting_config: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """
        Get the best classification for a set of tokens using dynamic priority-based weighting.
        Handles "No Match" cases when similarity is below threshold.
        
        Args:
            tokens: Ordered detection tokens (index 0 = highest priority)
            min_similarity: Minimum similarity threshold for valid classification
            use_dynamic_weighting: Enable dynamic token weighting system
            weighting_config: Configuration for weighting algorithm
            
        Returns:
            Dict with complete metadata or None for "No Match" cases
        """
        if not tokens:
            logger.debug("No tokens provided - returning None")
            return None
        
        # Configure dynamic weighting parameters
        if weighting_config is None:
            weighting_config = {
                'max_tokens': 10,
                'decay_factor': 0.1,
                'min_weight': 0.1,
                'hardware_boost': 1.5
            }
        
        # Assess token quality using dynamic weighting
        token_quality = self.assess_token_quality(tokens)
        
        # Apply stricter filtering based on token quality
        quality_threshold = 0.2  # Minimum quality score required
        if token_quality['quality_score'] < quality_threshold:
            logger.debug(f"Token quality too low: {token_quality['quality_score']:.3f} < {quality_threshold} - No Match")
            return None
        
        # Apply dynamic weighting to understand token priorities
        if use_dynamic_weighting:
            weighted_tokens = self.apply_dynamic_token_weighting(
                tokens,
                max_tokens=weighting_config['max_tokens'],
                decay_factor=weighting_config['decay_factor'],
                min_weight=weighting_config['min_weight']
            )
            
            if not weighted_tokens:
                logger.debug("No valid weighted tokens - No Match")
                return None
                
            logger.debug(f"Dynamic weighting: {len(weighted_tokens)} valid tokens from {len(tokens)} total")
        
        # Normalize tokens (optional) before searches
        tokens_norm = self.normalizer.normalize_tokens(tokens) if self.normalize_tokens else tokens

        # Vector DB is the source of truth for classification
        similar_items = self.search_similar_items(
            tokens_norm, k=3, use_dynamic_weighting=use_dynamic_weighting, 
            weighting_config=weighting_config
        )
        vector_match = None
        if similar_items and similar_items[0]['similarity_score'] >= min_similarity:
            vector_match = similar_items[0]

        # Itemtypes are used only to provide official naming (QA); do not decide ID
        # Use a slightly stricter threshold for QA naming than classification
        itemtype_match = self.find_itemtypes_match(
            tokens_norm, self.official_threshold, use_dynamic_weighting=use_dynamic_weighting
        )
        
        # Apply dynamic similarity thresholds based on token quality and context
        if token_quality['quality_score'] < 0.4:
            # For low quality tokens, require higher similarity
            adjusted_min_similarity = min_similarity * 1.2
            logger.debug(f"Low token quality, increasing threshold to {adjusted_min_similarity:.3f}")
        elif token_quality['has_hardware_terms'] and use_dynamic_weighting and weighted_tokens:
            # For hardware-relevant tokens with good weighting, allow slightly lower similarity
            hardware_token_ratio = sum(1 for wt in weighted_tokens if wt['is_hardware']) / len(weighted_tokens)
            if hardware_token_ratio > 0.5:  # More than 50% hardware tokens
                adjusted_min_similarity = min_similarity * 0.9
                logger.debug(f"High hardware relevance, decreasing threshold to {adjusted_min_similarity:.3f}")
            else:
                adjusted_min_similarity = min_similarity
        else:
            adjusted_min_similarity = min_similarity
        
        # Re-evaluate matches with adjusted threshold for "No Match" detection
        if vector_match and vector_match['similarity_score'] < adjusted_min_similarity:
            logger.debug(f"Vector match below threshold: {vector_match['similarity_score']:.3f} < {adjusted_min_similarity:.3f}")
            vector_match = None
        if itemtype_match and itemtype_match['similarity_score'] < adjusted_min_similarity:
            itemtype_match = None

        # If there is no vector DB match, return None to signal "No Match"
        if not vector_match:
            return None
        
        # Determine primary match based on similarity scores and dynamic weighting
        primary_source = None
        if itemtype_match and vector_match:
            # Prioritize itemtypes for better naming consistency
            itemtype_score = itemtype_match['similarity_score']
            vector_score = vector_match['similarity_score']
            
            # Apply hardware boost if dynamic weighting shows strong hardware context
            if use_dynamic_weighting and weighted_tokens:
                hardware_weight_sum = sum(wt['final_weight'] for wt in weighted_tokens if wt['is_hardware'])
                if hardware_weight_sum > 2.0:  # Strong hardware context
                    itemtype_score *= 1.1  # Slight boost for itemtypes naming
            
            # Prefer itemtypes if scores are close (within 10%)
            if abs(itemtype_score - vector_score) < 0.1:
                primary_source = 'itemtypes'
            elif itemtype_score >= vector_score:
                primary_source = 'itemtypes'
            else:
                primary_source = 'vector_database'
        elif itemtype_match:
            primary_source = 'itemtypes'
        elif vector_match:
            primary_source = 'vector_database'
        
        # Build comprehensive result: always use vector DB for classification
        original_id = vector_match['id']
        original_metadata = self.item_lookup.get(original_id, {})

        # Build result from vector DB match (classification)
        result = {
            'id': original_id,
            'reference_name': vector_match['name'],
            # Display-only QA fields are added separately and never used for display
            'category': vector_match.get('category', ''),
            'model': vector_match.get('model', ''),
            'brand': '',  # database metadata does not include brand
            'similarity_score': vector_match['similarity_score'],
            'source': 'vector_database',
            'contextual_text': original_metadata.get('contextual_text', ''),
            'vector_index': original_metadata.get('vector_index', None),
            'embedding_norm': original_metadata.get('embedding_norm', None),
            'legacyId': original_metadata.get('legacyId', None)
        }

        # Attach QA naming info (not for display)
        if itemtype_match:
            result['official_name'] = itemtype_match['name']
            result['official_similarity'] = float(itemtype_match.get('similarity_score', 0.0))
            # Optional consistency check (brand/family alignment)
            def _infer_brand(s: str) -> str:
                st = (s or '').lower()
                if any(x in st for x in ['nintendo', '任天堂', 'gamecube', 'wii', 'snes', 'n64', 'ds', 'switch', 'game boy']):
                    return 'nintendo'
                if any(x in st for x in ['playstation', 'sony', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'psp', 'vita']):
                    return 'sony'
                if any(x in st for x in ['xbox', 'microsoft']):
                    return 'microsoft'
                if 'sega' in st or 'saturn' in st or 'genesis' in st or 'megadrive' in st:
                    return 'sega'
                return ''
            if self.enforce_official_consistency:
                ref_brand = _infer_brand(result['reference_name'])
                off_brand = _infer_brand(itemtype_match['name'])
                result['official_consistency'] = (ref_brand == '' or off_brand == '' or ref_brand == off_brand)
            else:
                result['official_consistency'] = True
        else:
            result['official_name'] = None
            result['official_similarity'] = 0.0
            result['official_consistency'] = False
        
        # Add dynamic weighting analysis metadata to result
        if use_dynamic_weighting and weighted_tokens:
            result['dynamic_weighting_metadata'] = {
                'total_weighted_tokens': len(weighted_tokens),
                'hardware_tokens_count': sum(1 for wt in weighted_tokens if wt['is_hardware']),
                'top_5_tokens': [wt['token'] for wt in weighted_tokens[:5]],
                'highest_weight': weighted_tokens[0]['final_weight'],
                'hardware_weight_sum': sum(wt['final_weight'] for wt in weighted_tokens if wt['is_hardware']),
                'weighting_config': weighting_config
            }
        
        # Add token quality assessment metadata
        result['token_quality_metadata'] = {
            'quality_score': token_quality['quality_score'],
            'has_hardware_terms': token_quality['has_hardware_terms'],
            'hardware_ratio': token_quality['hardware_ratio'],
            'valid_token_count': token_quality['valid_token_count'],
            'adjusted_min_similarity': adjusted_min_similarity
        }
        
        logger.debug(
            f"Classification successful: {result.get('reference_name')} (sim: {result['similarity_score']:.3f}); "
            f"official_name: {result.get('official_name')} (sim: {result.get('official_similarity', 0.0):.3f}, "
            f"consistent: {result.get('official_consistency')})"
        )
        
        return result
    
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
                'vector_db_version': '2.1',
                'model_used': str(self.model_path),
                'min_similarity_threshold': 0.5,
                'dynamic_weighting_enabled': True,
                'weighting_algorithm': 'exponential_decay_with_priority',
                'weighting_config': {
                    'max_tokens': 10,
                    'decay_factor': 0.1,
                    'min_weight': 0.1,
                    'hardware_boost': 1.5
                },
                'features': [
                    'dynamic_token_prioritization',
                    'no_match_detection',
                    'hardware_context_boost',
                    'quality_based_threshold_adjustment'
                ]
            }
            
            # Add gdino_tokens at the end
            enhanced_data['gdino_tokens'] = data.get('gdino_tokens', {})
            
            # Process each detection
            gdino_tokens = data.get('gdino_tokens', {})
            
            for detection_id, tokens in gdino_tokens.items():
                if not tokens or not isinstance(tokens, list):
                    # No tokens available - mark as no match
                    enhanced_data['gdino_improved'][detection_id] = {
                        'id': 'No Match',
                        'reference_name': 'No Match',
                        'category': '',
                        'model': '',
                        'brand': '',
                        'source': 'none',
                        'contextual_text': '',
                        'vector_index': None,
                        'embedding_norm': None,
                        'legacyId': None,
                        'similarity_score': 0.0,
                        # QA-only fields even for no-match (clear values)
                        'official_name': None,
                        'official_similarity': 0.0,
                        'official_consistency': False
                    }
                    enhanced_data['gdino_improved_readable'][detection_id] = "No Match"
                    enhanced_data['gdino_similarity_scores'][detection_id] = 0.0
                    continue
                
                # Get best classification using dynamic weighting system
                weighting_config = {
                    'max_tokens': 10,  # Consider top 10 priority tokens
                    'decay_factor': 0.1,  # Exponential decay rate
                    'min_weight': 0.1,  # Minimum weight threshold
                    'hardware_boost': 1.5  # Boost for hardware-related tokens
                }
                
                classification_result = self.get_best_classification(
                    tokens,
                    min_similarity=0.55,
                    use_dynamic_weighting=True,
                    weighting_config=weighting_config
                )
                
                if classification_result:
                    # Store comprehensive metadata in gdino_improved
                    enhanced_data['gdino_improved'][detection_id] = {
                        'id': classification_result['id'],
                        'reference_name': classification_result.get('reference_name'),
                        'category': classification_result['category'],
                        'model': classification_result['model'],
                        'brand': classification_result['brand'],
                        'source': classification_result['source'],
                        'contextual_text': classification_result.get('contextual_text', ''),
                        'vector_index': classification_result.get('vector_index'),
                        'embedding_norm': classification_result.get('embedding_norm'),
                        'legacyId': classification_result.get('legacyId'),
                        'similarity_score': classification_result['similarity_score'],
                        # QA-only fields
                        'official_name': classification_result.get('official_name'),
                        'official_similarity': classification_result.get('official_similarity', 0.0),
                        'official_consistency': classification_result.get('official_consistency', False)
                    }
                    
                    # Add secondary matches if available
                    if classification_result.get('vector_reference_id'):
                        enhanced_data['gdino_improved'][detection_id]['vector_reference_id'] = classification_result['vector_reference_id']
                        enhanced_data['gdino_improved'][detection_id]['vector_reference_similarity'] = classification_result['vector_reference_similarity']
                    
                    # Set readable strictly from items.json classification
                    readable_name = classification_result.get('reference_name')
                    if readable_name == 'No Match' or not readable_name:
                        enhanced_data['gdino_improved_readable'][detection_id] = "No Match"
                    else:
                        enhanced_data['gdino_improved_readable'][detection_id] = readable_name
                    enhanced_data['gdino_similarity_scores'][detection_id] = round(classification_result['similarity_score'], 4)
                    # Mirror to enhanced_classification map if enabled
                    if self.write_enhanced_classification:
                        enhanced_data.setdefault('enhanced_classification', {})
                        enhanced_data['enhanced_classification'][detection_id] = enhanced_data['gdino_improved'][detection_id]
                    
                else:
                    # No good match found - create proper "No Match" entry
                    enhanced_data['gdino_improved'][detection_id] = {
                        'id': 'No Match',
                        'reference_name': 'No Match',
                        'category': '',
                        'model': '',
                        'brand': '',
                        'source': 'none',
                        'contextual_text': '',
                        'vector_index': None,
                        'embedding_norm': None,
                        'legacyId': None,
                        'similarity_score': 0.0,
                        'official_name': None,
                        'official_similarity': 0.0,
                        'official_consistency': False
                    }
                    enhanced_data['gdino_improved_readable'][detection_id] = "No Match"
                    enhanced_data['gdino_similarity_scores'][detection_id] = 0.0
                    if self.write_enhanced_classification:
                        enhanced_data.setdefault('enhanced_classification', {})
                        enhanced_data['enhanced_classification'][detection_id] = enhanced_data['gdino_improved'][detection_id]
            
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
                                
                                # Track category improvements from gdino_improved nested data
                                improved_data = enhanced_data.get('gdino_improved', {}).get(detection_id, {})
                                category = improved_data.get('category', 'Unknown')
                                if category and category != '':
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


def main(argv: Optional[List[str]] = None):
    """CLI for enhancing GroundingDINO results."""
    parser = argparse.ArgumentParser(description="Enhance GDINO results with vector DB")
    parser.add_argument("--vector-db", default=os.getenv("STRUSTORE_DB_PATH", "models/vector_database"))
    parser.add_argument("--model", default=os.getenv("STRUSTORE_MODEL_PATH", "intfloat/multilingual-e5-base"))
    parser.add_argument("--gdino-dir", default=os.getenv("STRUSTORE_GDINO_DIR", "../gdinoOutput/final"))
    parser.add_argument("--itemtypes", default=os.getenv("STRUSTORE_ITEMTYPES_PATH", "../itemtypes.json"))
    parser.add_argument("--no-normalize", action="store_true", help="Disable token normalization")
    parser.add_argument("--official-threshold", type=float, default=float(os.getenv("STRUSTORE_OFFICIAL_THRESHOLD", 0.60)), help="Threshold for accepting itemtypes QA name")
    parser.add_argument("--no-consistency-gate", action="store_true", help="Disable brand/family consistency check for official QA name")
    parser.add_argument("--no-enhanced-classification", action="store_true", help="Do not write enhanced_classification map")
    parser.add_argument("--log-level", default=os.getenv("STRUSTORE_LOG_LEVEL", "INFO"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    enhancer = GdinoResultEnhancer(
        vector_db_path=args.vector_db,
        model_path=args.model,
        gdino_output_dir=args.gdino_dir,
        itemtypes_path=args.itemtypes,
        normalize_tokens=(not args.no_normalize),
        itemtypes_display_only=True,
        write_enhanced_classification=(not args.no_enhanced_classification),
        official_threshold=args.official_threshold,
        enforce_official_consistency=(not args.no_consistency_gate),
    )

    enhancer.load_vector_database()
    enhancer.load_itemtypes_database()
    results = enhancer.process_all_files(dry_run=args.dry_run)
    return results


if __name__ == '__main__':
    results = main()
    print(f"✅ Enhancement completed: {results}")

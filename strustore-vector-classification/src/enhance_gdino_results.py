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
                 itemtypes_path: str = "../../itemtypes.json"):
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
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.itemtypes_embeddings.astype('float32'))
            self.itemtypes_index.add(self.itemtypes_embeddings.astype('float32'))
            
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
            
            # Create search query
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
            query_embedding = self.model.encode([query_text])
            faiss.normalize_L2(query_embedding.astype('float32'))
            
            # Search
            scores, indices = self.itemtypes_index.search(query_embedding.astype('float32'), k)
            
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

    def get_best_classification(self, tokens: List[str], min_similarity: float = 0.5, 
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
        
        # Search both databases with dynamic weighting
        itemtype_match = self.find_itemtypes_match(
            tokens, min_similarity, use_dynamic_weighting=use_dynamic_weighting
        )
        
        similar_items = self.search_similar_items(
            tokens, k=3, use_dynamic_weighting=use_dynamic_weighting, 
            weighting_config=weighting_config
        )
        vector_match = None
        if similar_items and similar_items[0]['similarity_score'] >= min_similarity:
            vector_match = similar_items[0]
        
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
        if itemtype_match and itemtype_match['similarity_score'] < adjusted_min_similarity:
            logger.debug(f"Itemtype match below threshold: {itemtype_match['similarity_score']:.3f} < {adjusted_min_similarity:.3f}")
            itemtype_match = None
        if vector_match and vector_match['similarity_score'] < adjusted_min_similarity:
            logger.debug(f"Vector match below threshold: {vector_match['similarity_score']:.3f} < {adjusted_min_similarity:.3f}")
            vector_match = None
        
        # If no matches meet the threshold, return None for "No Match"
        if not itemtype_match and not vector_match:
            logger.debug(f"No matches above similarity threshold {adjusted_min_similarity:.3f} - No Match")
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
        
        # Build comprehensive result with improved naming logic
        if primary_source == 'itemtypes':
            # Primary match is itemtypes
            result = {
                'id': itemtype_match['id'],
                'reference_name': None,
                'itemtypes_name': itemtype_match['name'],
                'category': itemtype_match['category'],
                'model': itemtype_match.get('console', ''),
                'brand': itemtype_match.get('brand', ''),
                'similarity_score': itemtype_match['similarity_score'],
                'source': 'itemtypes',
                'contextual_text': itemtype_match.get('contextual_text', ''),
                'vector_index': None,
                'embedding_norm': None,
                'legacyId': None
            }
            
            # Try to find corresponding vector database match for reference name
            if vector_match:
                result['reference_name'] = vector_match['name']
                result['vector_reference_id'] = vector_match['id']
                result['vector_reference_similarity'] = vector_match['similarity_score']
            
        else:
            # Primary match is vector database
            original_id = vector_match['id']
            original_metadata = self.item_lookup.get(original_id, {})
            
            result = {
                'id': original_id,
                'reference_name': vector_match['name'],
                'itemtypes_name': None,  # Will be set below if itemtype_match exists
                'category': vector_match['category'],
                'model': vector_match.get('model', ''),
                'brand': '',  # Original database doesn't have brand field
                'similarity_score': vector_match['similarity_score'],
                'source': 'vector_database',
                'contextual_text': original_metadata.get('contextual_text', ''),
                'vector_index': original_metadata.get('vector_index', None),
                'embedding_norm': original_metadata.get('embedding_norm', None),
                'legacyId': original_metadata.get('legacyId', None)
            }
            
            # Try to find corresponding itemtypes match for enhanced naming
            if itemtype_match:
                result['itemtypes_name'] = itemtype_match['name']
                result['itemtypes_similarity'] = itemtype_match['similarity_score']
                # Update brand if itemtypes has better brand info
                if itemtype_match.get('brand'):
                    result['brand'] = itemtype_match['brand']
            else:
                # Try to search for itemtypes match with lower threshold
                fallback_itemtype = self.find_itemtypes_match(
                    tokens, min_similarity * 0.8, use_dynamic_weighting=use_dynamic_weighting
                )
                if fallback_itemtype:
                    result['itemtypes_name'] = fallback_itemtype['name']
                    result['itemtypes_similarity'] = fallback_itemtype['similarity_score']
                    if fallback_itemtype.get('brand'):
                        result['brand'] = fallback_itemtype['brand']
        
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
        
        logger.debug(f"Classification successful: {result.get('itemtypes_name') or result.get('reference_name')} "
                    f"(similarity: {result['similarity_score']:.3f}, source: {result['source']})")
        
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
                        'itemtypes_name': None,
                        'category': '',
                        'model': '',
                        'brand': '',
                        'source': 'none',
                        'contextual_text': '',
                        'vector_index': None,
                        'embedding_norm': None,
                        'legacyId': None,
                        'similarity_score': 0.0
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
                    min_similarity=0.5, 
                    use_dynamic_weighting=True,
                    weighting_config=weighting_config
                )
                
                if classification_result:
                    # Store comprehensive metadata in gdino_improved
                    enhanced_data['gdino_improved'][detection_id] = {
                        'id': classification_result['id'],
                        'reference_name': classification_result.get('reference_name'),
                        'itemtypes_name': classification_result.get('itemtypes_name'),
                        'category': classification_result['category'],
                        'model': classification_result['model'],
                        'brand': classification_result['brand'],
                        'source': classification_result['source'],
                        'contextual_text': classification_result.get('contextual_text', ''),
                        'vector_index': classification_result.get('vector_index'),
                        'embedding_norm': classification_result.get('embedding_norm'),
                        'legacyId': classification_result.get('legacyId'),
                        'similarity_score': classification_result['similarity_score']
                    }
                    
                    # Add secondary matches if available
                    if classification_result.get('vector_reference_id'):
                        enhanced_data['gdino_improved'][detection_id]['vector_reference_id'] = classification_result['vector_reference_id']
                        enhanced_data['gdino_improved'][detection_id]['vector_reference_similarity'] = classification_result['vector_reference_similarity']
                    
                    if classification_result.get('itemtypes_similarity'):
                        enhanced_data['gdino_improved'][detection_id]['itemtypes_similarity'] = classification_result['itemtypes_similarity']
                    
                    # Set readable name with priority: itemtypes_name > reference_name
                    readable_name = classification_result.get('itemtypes_name') or classification_result.get('reference_name')
                    if readable_name:
                        # Show both names if both exist
                        if classification_result.get('itemtypes_name') and classification_result.get('reference_name'):
                            if classification_result.get('itemtypes_name') != classification_result.get('reference_name'):
                                readable_name = f"{classification_result['itemtypes_name']} ({classification_result['reference_name']})"
                    
                    enhanced_data['gdino_improved_readable'][detection_id] = readable_name or "Unknown"
                    enhanced_data['gdino_similarity_scores'][detection_id] = round(classification_result['similarity_score'], 4)
                    
                else:
                    # No good match found - create proper "No Match" entry
                    enhanced_data['gdino_improved'][detection_id] = {
                        'id': 'No Match',
                        'reference_name': 'No Match',
                        'itemtypes_name': None,
                        'category': '',
                        'model': '',
                        'brand': '',
                        'source': 'none',
                        'contextual_text': '',
                        'vector_index': None,
                        'embedding_norm': None,
                        'legacyId': None,
                        'similarity_score': 0.0
                    }
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


def main():
    """Main function to enhance GroundingDINO results."""
    
    # Configuration - adapt paths based on current directory
    model_path = "intfloat/multilingual-e5-base" 
    
    # Check if we're running from strustore or strustore-vector-classification directory
    import os
    current_dir = os.getcwd()
    if current_dir.endswith('strustore-vector-classification'):
        vector_db_path = "models/vector_database"
        gdino_output_dir = "../gdinoOutput/final"
        itemtypes_path = "../itemtypes.json"
    else:
        # Running from strustore directory
        vector_db_path = "strustore-vector-classification/models/vector_database"
        gdino_output_dir = "gdinoOutput/final"
        itemtypes_path = "itemtypes.json"
    
    # Debug: Check if paths exist
    from pathlib import Path
    gdino_path = Path(gdino_output_dir)
    vector_path = Path(vector_db_path)
    itemtypes_file = Path(itemtypes_path)
    
    logger.info(f"🔍 Current directory: {current_dir}")
    logger.info(f"🔍 GDINO path '{gdino_path}' exists: {gdino_path.exists()}")
    logger.info(f"🔍 Vector DB path '{vector_path}' exists: {vector_path.exists()}")
    logger.info(f"🔍 Itemtypes path '{itemtypes_file}' exists: {itemtypes_file.exists()}")
    
    if gdino_path.exists():
        json_files = list(gdino_path.rglob("*.json"))
        logger.info(f"🔍 Found {len(json_files)} JSON files in GDINO directory")
    
    logger.info("🚀 Starting GroundingDINO Result Enhancement Pipeline with Itemtypes Support")
    
    # Initialize enhancer
    enhancer = GdinoResultEnhancer(vector_db_path, model_path, gdino_output_dir, itemtypes_path)
    
    # Load vector database
    enhancer.load_vector_database()
    
    # Load itemtypes database
    enhancer.load_itemtypes_database()
    
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
    # Run the enhancement process
    enhancer = GdinoResultEnhancer()
    enhancer.load_vector_database()
    results = enhancer.process_all_files()
    print(f"✅ Enhancement completed: {results}")

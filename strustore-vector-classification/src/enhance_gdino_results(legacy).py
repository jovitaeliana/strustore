"""
GroundingDINO Result Enhancement Pipeline

This script enhances existing GroundingDINO detection results by using the vector database
for improved semantic classification. It processes all JSON files in gdinoOutput/final/
and adds vector database classifications while preserving original results for comparison.
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
from collections import defaultdict, Counter

# Simplified classification system with defensive programming and config-based terms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
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


class CategoryDetector:
    """
    Industry-grade category detection using learned embeddings instead of hardcoded keywords.
    Automatically learns category boundaries from vector database content.
    """
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.category_centroids = {}
        self.category_boundaries = {}
        self.token_embeddings_cache = {}
        self.is_trained = False
        
    def train_from_database(self, metadata: List[Dict], itemtypes_metadata: List[Dict] = None):
        """
        Learn category embeddings from existing database content.
        
        Args:
            metadata: Vector database metadata
            itemtypes_metadata: Itemtypes database metadata (optional)
        """
        logger.info("🤖 Training category detector from database content...")
        
        # Combine both databases
        all_items = metadata.copy()
        if itemtypes_metadata:
            all_items.extend(itemtypes_metadata)
        
        # Group items by category
        category_items = defaultdict(list)
        category_tokens = defaultdict(list)
        
        for item in all_items:
            category = item.get('category', 'Unknown')
            name = item.get('name', '')
            contextual_text = item.get('contextual_text', '')
            
            # Extract meaningful tokens from name and context
            tokens = self._extract_tokens(name, contextual_text)
            category_items[category].append(item)
            category_tokens[category].extend(tokens)
        
        # Compute category centroids
        for category, tokens in category_tokens.items():
            if len(tokens) < 3:  # Skip categories with too few tokens
                continue
            
            # Get embeddings for tokens
            unique_tokens = list(set(tokens))
            embeddings = self.model.encode(unique_tokens)
            
            # Compute centroid
            centroid = np.mean(embeddings, axis=0)
            self.category_centroids[category] = {
                'centroid': centroid,
                'tokens': unique_tokens,
                'item_count': len(category_items[category])
            }
            
            logger.debug(f"Category '{category}': {len(unique_tokens)} unique tokens, {len(category_items[category])} items")
        
        # Compute category boundaries (distance thresholds)
        self._compute_category_boundaries()
        
        self.is_trained = True
        logger.info(f"✅ Category detector trained on {len(self.category_centroids)} categories")
        
    def _extract_tokens(self, name: str, contextual_text: str) -> List[str]:
        """Extract meaningful tokens from item name and context."""
        text = f"{name} {contextual_text}".lower()
        
        # Basic tokenization and filtering
        tokens = []
        for word in text.split():
            # Clean token
            token = word.strip('.,!?()[]{}"\'')
            
            # Filter criteria
            if (len(token) >= 2 and 
                not token.isdigit() and 
                token not in {'the', 'and', 'or', 'for', 'with', 'from', 'to', 'of', 'in', 'on', 'at'}):
                tokens.append(token)
        
        return tokens
    
    def _compute_category_boundaries(self):
        """Compute similarity thresholds for each category with negative sampling."""
        categories = list(self.category_centroids.keys())
        
        # Create negative examples (non-gaming terms) for better boundaries
        negative_terms = [
            'mouse', 'mice', 'bike', 'bicycle', 'car', 'auto', 'vehicle', 'truck',
            'clock', 'watch', 'time', 'alarm', 'phone', 'mobile', 'cell', 'smartphone',
            'book', 'magazine', 'paper', 'pen', 'pencil', 'office', 'desk', 'chair',
            'kitchen', 'cooking', 'food', 'drink', 'coffee', 'tea', 'water', 'juice',
            'clothes', 'shirt', 'pants', 'shoes', 'fashion', 'style', 'clothing',
            'furniture', 'table', 'bed', 'room', 'house', 'home', 'building',
            'tool', 'hammer', 'screwdriver', 'wrench', 'drill', 'saw', 'equipment',
            'camera', 'photo', 'picture', 'lens', 'tripod', 'flash', 'photography'
        ]
        
        # Get embeddings for negative examples
        logger.debug(f"Computing boundaries with {len(negative_terms)} negative examples")
        negative_embeddings = self.model.encode(negative_terms)
        
        for category in categories:
            centroid = self.category_centroids[category]['centroid']
            
            # Compute similarities to negative examples
            negative_similarities = cosine_similarity([centroid], negative_embeddings)[0]
            max_negative_similarity = np.max(negative_similarities)
            avg_negative_similarity = np.mean(negative_similarities)
            
            # Compute distances to other gaming category centroids
            gaming_distances = []
            for other_category in categories:
                if other_category != category:
                    other_centroid = self.category_centroids[other_category]['centroid']
                    distance = cosine_similarity([centroid], [other_centroid])[0][0]
                    gaming_distances.append(distance)
            
            # Set boundary to be stricter than the worst negative example
            if gaming_distances:
                min_gaming_distance = min(gaming_distances) if gaming_distances else 0.8
                # Boundary should be higher than negative examples but lower than gaming categories
                boundary = max(
                    max_negative_similarity + 0.1,  # Above worst negative example
                    avg_negative_similarity + 0.2,  # Well above average negative
                    min_gaming_distance * 0.7       # Conservative vs other gaming categories
                )
            else:
                # Single category - be very strict
                boundary = max(max_negative_similarity + 0.15, 0.6)
            
            self.category_boundaries[category] = min(boundary, 0.85)  # Cap at 0.85
            
            logger.debug(f"Category '{category}': boundary={boundary:.3f}, "
                        f"max_negative={max_negative_similarity:.3f}, "
                        f"avg_negative={avg_negative_similarity:.3f}")
            
    def detect_token_categories(self, tokens: List[str], top_k: int = 3) -> Dict[str, Any]:
        """
        Detect categories for given tokens using learned embeddings.
        
        Args:
            tokens: List of tokens to analyze
            top_k: Number of top categories to return
            
        Returns:
            Dict with category predictions and confidence scores
        """
        if not self.is_trained:
            logger.warning("CategoryDetector not trained! Using fallback detection.")
            return self._fallback_detection(tokens)
        
        if not tokens:
            return {'categories': [], 'confidence': 0.0, 'is_gaming': False}
        
        # Get embeddings for tokens
        token_embeddings = []
        valid_tokens = []
        
        for token in tokens[:20]:  # Process top 20 tokens for better evaluation
            token_clean = token.lower().strip()
            if len(token_clean) >= 2:
                if token_clean not in self.token_embeddings_cache:
                    self.token_embeddings_cache[token_clean] = self.model.encode([token_clean])[0]
                token_embeddings.append(self.token_embeddings_cache[token_clean])
                valid_tokens.append(token_clean)
        
        if not token_embeddings:
            return {'categories': [], 'confidence': 0.0, 'is_gaming': False}
        
        # Compute average token embedding
        query_embedding = np.mean(token_embeddings, axis=0).reshape(1, -1)
        
        # Compute similarities to category centroids
        category_scores = {}
        for category, info in self.category_centroids.items():
            centroid = info['centroid'].reshape(1, -1)
            similarity = cosine_similarity(query_embedding, centroid)[0][0]
            boundary = self.category_boundaries[category]
            
            # Only include if above boundary threshold
            if similarity >= boundary:
                category_scores[category] = {
                    'similarity': similarity,
                    'boundary': boundary,
                    'confidence': (similarity - boundary) / (1.0 - boundary) if boundary < 1.0 else similarity
                }
        
        # Sort by similarity
        sorted_categories = sorted(category_scores.items(), 
                                 key=lambda x: x[1]['similarity'], reverse=True)[:top_k]
        
        # Determine if gaming-related with stricter criteria
        gaming_categories = {
            'Video Game Consoles', 'Controllers & Attachments', 'Video Games', 
            'Memory Cards & Expansion Packs', 'Other Video Game Accessories',
            'Cables & Adapters'  # Include if gaming-related cables
        }
        
        max_confidence = max([score['confidence'] for _, score in sorted_categories]) if sorted_categories else 0.0
        
        # More restrictive gaming detection
        is_gaming = False
        if sorted_categories:
            top_category, top_score_dict = sorted_categories[0]
            top_similarity = top_score_dict['similarity']
            # Must be gaming category AND have reasonable similarity AND confidence
            is_gaming = (
                top_category in gaming_categories and 
                top_similarity >= 0.6 and  # Balanced similarity threshold
                max_confidence >= 0.2       # Lower confidence threshold
            )
        
        # Additional validation: check if tokens suggest non-gaming items
        non_gaming_indicators = {
            'mouse', 'mice', 'bike', 'bicycle', 'car', 'auto', 'vehicle',
            'clock', 'watch', 'phone', 'mobile', 'furniture', 'chair', 'table'
        }
        
        # If any token is clearly non-gaming, reduce gaming confidence
        non_gaming_token_count = sum(1 for token in valid_tokens 
                                   if any(ng in token.lower() for ng in non_gaming_indicators))
        
        if non_gaming_token_count > 0:
            # Penalize gaming confidence if non-gaming tokens present
            penalty_factor = 1.0 - (non_gaming_token_count / len(valid_tokens))
            max_confidence *= penalty_factor
            is_gaming = is_gaming and penalty_factor > 0.5
        
        return {
            'categories': [(cat, score['similarity']) for cat, score in sorted_categories],
            'confidence': max_confidence,
            'is_gaming': is_gaming,
            'valid_tokens': valid_tokens,
            'gaming_categories': gaming_categories,
            'non_gaming_penalty': non_gaming_token_count / len(valid_tokens) if valid_tokens else 0.0
        }
    
    def _fallback_detection(self, tokens: List[str]) -> Dict[str, Any]:
        """Improved fallback detection with stricter gaming criteria."""
        gaming_indicators = {
            'nintendo', 'playstation', 'xbox', 'switch', 'controller', 'console',
            'ds', 'dsi', 'wii', 'gameboy', 'gaming', 'handheld', 'joystick', 'gamepad',
            'dualshock', 'joycon', 'pro', 'slim', 'lite'
        }
        
        strong_non_gaming_indicators = {
            'mouse', 'mice', 'bike', 'bicycle', 'car', 'auto', 'vehicle', 'truck',
            'phone', 'mobile', 'cell', 'smartphone', 'clock', 'watch', 'alarm',
            'chair', 'table', 'desk', 'furniture', 'book', 'kitchen', 'cooking'
        }
        
        # Count exact matches
        gaming_count = sum(1 for token in tokens[:20] 
                          if any(token.lower() == g or g in token.lower() for g in gaming_indicators))
        
        strong_non_gaming_count = sum(1 for token in tokens[:20] 
                                    if any(token.lower() == ng or ng in token.lower() for ng in strong_non_gaming_indicators))
        
        # Much stricter criteria for gaming classification
        is_gaming = (
            gaming_count > 0 and                    # Must have gaming indicators
            strong_non_gaming_count == 0 and        # No strong non-gaming indicators
            gaming_count >= len(tokens[:20]) * 0.3  # At least 30% gaming tokens
        )
        
        confidence = (gaming_count / len(tokens[:20])) if tokens else 0.0
        
        # Penalize if non-gaming indicators present
        if strong_non_gaming_count > 0:
            confidence *= 0.1  # Heavy penalty
            is_gaming = False
        
        return {
            'categories': [('Gaming Hardware', confidence)] if is_gaming and confidence > 0.3 else [],
            'confidence': confidence if is_gaming else 0.0,
            'is_gaming': is_gaming,
            'valid_tokens': tokens[:20],
            'gaming_categories': {'Gaming Hardware'},
            'gaming_count': gaming_count,
            'non_gaming_count': strong_non_gaming_count
        }

class GdinoResultEnhancer:
    """
    Enhances GroundingDINO detection results using vector database semantic search.
    """
    
    def __init__(self, 
                 vector_db_path: str = "models/vector_database",
                 model_path: str = "intfloat/multilingual-e5-base",
                 gdino_output_dir: str = "../gdinoOutput/final",
                 itemtypes_path: str = "itemtypes.json"):
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
        self.category_detector = None
        
        # Dynamic weighting system is implemented within class methods
        
    def load_vector_database(self) -> None:
        """Load the vector database and model for semantic search."""
        try:
            logger.info(f"🔄 Loading vector database from: {self.vector_db_path}")
            
            # Load items.json for proper ID mapping
            items_json_path = Path("items.json")  # Located at project root
            if items_json_path.exists():
                logger.info(f"Loading items.json from: {items_json_path}")
                with open(items_json_path, 'r', encoding='utf-8') as f:
                    items_list = json.load(f)
                
                # Create ID-based lookup
                self.items_data = {}
                for item in items_list:
                    item_id = item.get('id')
                    if item_id:
                        self.items_data[item_id] = item
                
                logger.info(f"✅ Loaded {len(self.items_data)} items from items.json")
            else:
                logger.warning(f"items.json not found at {items_json_path} - ID mapping may be incomplete")
                self.items_data = {}
            
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
            
            # Initialize and train CategoryDetector
            self.category_detector = CategoryDetector(self.model)
                
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
            
            # Train CategoryDetector with both databases
            if self.category_detector and self.metadata:
                self.category_detector.train_from_database(self.metadata, itemtype_metadata)
            
        except Exception as e:
            logger.error(f"Failed to load itemtypes database: {e}")
            # Continue without itemtypes support
    
    def _validate_token_existence(self, tokens: List[str], match_name: str) -> bool:
        """
        Validate that key terms in the match name actually exist in the tokens.
        Prevents over-specification to non-existent terms.
        
        Args:
            tokens: Original gdino_tokens
            match_name: Name of the proposed match
            
        Returns:
            True if match is reasonable based on existing tokens
        """
        if not tokens or not match_name:
            return True
        
        tokens_lower = [t.lower().strip() for t in tokens[:15]]  # Check top 15 tokens
        match_lower = match_name.lower()
        
        # Extract key descriptive terms that should exist in tokens if they're in the match
        problematic_terms = {
            'guitar', 'freaks', 'dance', 'revolution', 'rockband', 'hero',
            'pokeball', 'pikachu', 'mario', 'zelda', 'sonic',
            'pro', 'slim', 'lite', 'oled', 'xl', 'sp',  # Model variants
            'wireless', 'wired', 'bluetooth',
            'memory', 'card', 'stick', 'adapter',
            'charger', 'cable', 'power', 'ac'
        }
        
        # Check if match contains specific terms that aren't in tokens
        for term in problematic_terms:
            if term in match_lower:
                # This specific term is in the match name - it should exist in tokens
                if not any(term in token_lower or token_lower in term for token_lower in tokens_lower):
                    logger.debug(f"Match '{match_name}' contains '{term}' but term not found in tokens - rejecting for accuracy")
                    return False
        
        return True
    
    def _calculate_token_coverage(self, tokens: List[str], target_text: str) -> float:
        """
        Calculate how many tokens from the gdino_tokens are covered in the target text.
        Higher coverage indicates better match quality.
        
        Args:
            tokens: List of gdino tokens (top 10 for efficiency)
            target_text: Text to check coverage against
            
        Returns:
            Coverage score between 0.0 and 1.0
        """
        if not tokens or not target_text:
            return 0.0
        
        target_lower = target_text.lower()
        coverage_count = 0
        
        for token in tokens[:10]:  # Check top 10 tokens
            token_clean = token.strip().lower()
            if len(token_clean) >= 2 and token_clean in target_lower:
                coverage_count += 1
        
        return coverage_count / min(10, len(tokens))

    def find_exact_matches(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Industry-level exact match detection with hierarchical token priority.
        Prioritizes exact matches over semantic similarity based on gdino_tokens position.
        Includes gaming-relevance validation to prevent false positives.
        
        Args:
            tokens: Ordered list of tokens from gdino_tokens (index 0 = highest priority)
            
        Returns:
            List of exact matches sorted by token hierarchy and match quality
        """
        if not tokens:
            return []
        
        # First assess if tokens represent gaming hardware
        token_quality = self.assess_token_quality(tokens)
        gaming_relevance = token_quality.get('gaming_relevance', 0.0)
        
        # If tokens have very low gaming relevance, skip exact matching to prevent false positives
        if gaming_relevance < 0.1:  # Very strict threshold for exact matches
            logger.debug(f"Tokens have low gaming relevance ({gaming_relevance:.3f}) - skipping exact match to prevent false positives")
            return []
        
        exact_matches = []
        
        # Search for exact matches using token hierarchy
        for token_idx, token in enumerate(tokens[:15]):  # Check top 15 tokens
            token_clean = token.strip().lower()
            if len(token_clean) < 2:
                continue
            
            # Search vector database for exact matches
            if self.metadata:
                for db_idx, item in enumerate(self.metadata):
                    item_name = item.get('name', '').lower()
                    contextual_text = item.get('contextual_text', '').lower()
                    
                    # Exact match in item name (highest priority) - but validate token coverage
                    if token_clean in item_name.split() and self._is_significant_match(token_clean, item_name, 'name'):
                        # Calculate token coverage score - how many tokens from gdino_tokens are present in this item
                        coverage_score = self._calculate_token_coverage(tokens, item_name + " " + item.get('contextual_text', ''))
                        
                        # Only accept exact matches with reasonable coverage (prevents "Pokèball" matching "switch")
                        if coverage_score >= 0.2:  # At least 20% of tokens should be covered
                            exact_matches.append({
                                'item': item,
                                'match_type': 'vector_name_exact',
                                'token_index': token_idx,
                                'token': token,
                                'matched_field': 'name',
                                'matched_text': item_name,
                                'hierarchy_score': 1000 - token_idx + (coverage_score * 200),  # Boost score based on coverage
                                'database_index': db_idx,
                                'source': 'vector_database',
                                'coverage_score': coverage_score
                            })
                    
                    # Exact match in contextual text (only for significant matches)
                    elif token_clean in contextual_text and self._is_significant_match(token_clean, contextual_text, 'context'):
                        # Calculate coverage for context matches too
                        coverage_score = self._calculate_token_coverage(tokens, item_name + " " + item.get('contextual_text', ''))
                        
                        if coverage_score >= 0.2:  # Same coverage requirement
                            exact_matches.append({
                                'item': item,
                                'match_type': 'vector_context_exact',
                                'token_index': token_idx,
                                'token': token,
                                'matched_field': 'contextual_text',
                                'matched_text': contextual_text,
                                'hierarchy_score': 500 - token_idx + (coverage_score * 150),  # Lower than name matches but with coverage boost
                                'database_index': db_idx,
                                'source': 'vector_database',
                                'coverage_score': coverage_score
                            })
            
            # Search itemtypes database for exact matches
            if hasattr(self, 'itemtypes_metadata') and self.itemtypes_metadata:
                for db_idx, item in enumerate(self.itemtypes_metadata):
                    item_name = item.get('name', '').lower()
                    contextual_text = item.get('contextual_text', '').lower()
                    
                    # Exact match in itemtype name (high priority)
                    if token_clean in item_name.split() and self._is_significant_match(token_clean, item_name, 'name'):
                        coverage_score = self._calculate_token_coverage(tokens, item_name + " " + item.get('contextual_text', ''))
                        
                        if coverage_score >= 0.2:
                            exact_matches.append({
                                'item': item,
                                'match_type': 'itemtype_name_exact',
                                'token_index': token_idx,
                                'token': token,
                                'matched_field': 'name',
                                'matched_text': item_name,
                                'hierarchy_score': 800 - token_idx + (coverage_score * 180),
                                'database_index': db_idx,
                                'source': 'itemtypes',
                                'coverage_score': coverage_score
                            })
                    
                    # Exact match in itemtype context (only for significant matches)
                    elif token_clean in contextual_text and self._is_significant_match(token_clean, contextual_text, 'context'):
                        coverage_score = self._calculate_token_coverage(tokens, item_name + " " + item.get('contextual_text', ''))
                        
                        if coverage_score >= 0.2:
                            exact_matches.append({
                                'item': item,
                                'match_type': 'itemtype_context_exact',
                                'token_index': token_idx,
                                'token': token,
                                'matched_field': 'contextual_text',
                                'matched_text': contextual_text,
                                'hierarchy_score': 400 - token_idx + (coverage_score * 120),
                                'database_index': db_idx,
                                'source': 'itemtypes',
                                'coverage_score': coverage_score
                            })
        
        # Find compound phrase matches (e.g., "nintendo ds")
        compound_matches = self._find_compound_phrase_matches(tokens)
        exact_matches.extend(compound_matches)
        
        # Sort by hierarchy score (higher = better, earlier tokens = higher priority)
        exact_matches.sort(key=lambda x: x['hierarchy_score'], reverse=True)
        
        return exact_matches
    
    def _find_compound_phrase_matches(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Find compound phrase matches like 'nintendo ds' using token hierarchy.
        
        Args:
            tokens: Ordered list of tokens from gdino_tokens
            
        Returns:
            List of compound matches with hierarchy scoring
        """
        if len(tokens) < 2:
            return []
        
        compound_matches = []
        
        # Generate compound phrases from consecutive tokens (respecting hierarchy)
        for i in range(min(8, len(tokens) - 1)):  # Check top 8 positions
            for j in range(i + 1, min(i + 4, len(tokens))):  # Max 3-token phrases
                phrase_tokens = tokens[i:j+1]
                phrase = ' '.join(phrase_tokens).lower().strip()
                
                if len(phrase) < 4:  # Skip very short phrases
                    continue
                
                # Search in both databases for compound phrase
                databases = [
                    ('vector_database', self.metadata or []),
                    ('itemtypes', getattr(self, 'itemtypes_metadata', []))
                ]
                
                for db_name, db_items in databases:
                    if not db_items:
                        continue
                        
                    for db_idx, item in enumerate(db_items):
                        item_name = item.get('name', '').lower()
                        contextual_text = item.get('contextual_text', '').lower()
                        
                        # Compound phrase match in name (very high priority)
                        if phrase in item_name:
                            compound_matches.append({
                                'item': item,
                                'match_type': f'{db_name}_compound_name',
                                'token_index': i,  # Use starting token index
                                'phrase': phrase,
                                'phrase_tokens': phrase_tokens,
                                'matched_field': 'name',
                                'matched_text': item_name,
                                # Compound matches get very high scores, earlier positions get higher scores
                                'hierarchy_score': 2000 - i * 10 + len(phrase_tokens) * 5,
                                'database_index': db_idx,
                                'source': db_name
                            })
                        
                        # Compound phrase match in context
                        elif phrase in contextual_text:
                            compound_matches.append({
                                'item': item,
                                'match_type': f'{db_name}_compound_context',
                                'token_index': i,
                                'phrase': phrase,
                                'phrase_tokens': phrase_tokens,
                                'matched_field': 'contextual_text',
                                'matched_text': contextual_text,
                                'hierarchy_score': 1500 - i * 8 + len(phrase_tokens) * 4,
                                'database_index': db_idx,
                                'source': db_name
                            })
        
        return compound_matches
    
    def _is_significant_match(self, token: str, target_text: str, field_type: str) -> bool:
        """
        Validate if a token match is significant enough to warrant exact match status.
        Prevents false positives from generic terms like "black", "electric", etc.
        
        Args:
            token: The token being matched
            target_text: The text field being matched against
            field_type: 'name' or 'context' to adjust validation strictness
            
        Returns:
            True if the match is significant, False if it's too generic
        """
        # Generic terms that shouldn't trigger exact matches (prevent false positives)
        generic_terms = {
            'black', 'white', 'red', 'blue', 'green', 'silver', 'gray', 'grey',
            'electric', 'wireless', 'battery', 'power', 'cable', 'cord', 'wire',
            'mouse', 'keyboard', 'phone', 'bike', 'car', 'clock', 'watch',
            'box', 'case', 'bag', 'stand', 'holder', 'mount', 'clip',
            'new', 'used', 'old', 'vintage', 'original', 'replica', 'copy',
            'small', 'large', 'mini', 'big', 'compact', 'portable', 'handheld',
            'working', 'tested', 'broken', 'repair', 'parts', 'spare',
            'free', 'shipping', 'fast', 'quick', 'cheap', 'expensive'
        }
        
        token_clean = token.lower().strip()
        
        # Skip very generic terms unless they're in a gaming context
        if token_clean in generic_terms:
            # For name matches, be more permissive (colors/descriptors can be part of product names)
            if field_type == 'name':
                # Allow if the token is a small part of a longer, more specific name
                name_words = target_text.lower().split()
                return len(name_words) >= 3 and any(hw in target_text.lower() for hw in 
                    ['nintendo', 'playstation', 'xbox', 'controller', 'console', 'ds', 'switch'])
            else:
                # For context matches, be very strict with generic terms
                return False
        
        # Skip very short tokens unless they're known gaming abbreviations
        if len(token_clean) <= 2:
            gaming_abbrevs = {'ds', 'gc', 'ps', 'xbox', 'wii', 'gba', 'psp', 'pro', 'xl', 'sp'}
            return token_clean in gaming_abbrevs
        
        # Allow longer, more specific terms
        return True

    def apply_hierarchical_token_priority(self, tokens: List[str], max_tokens: int = 10) -> List[Dict[str, Any]]:
        """
        Apply pure hierarchical token priority based on gdino_tokens position.
        Industry-level approach: position in gdino_tokens determines priority, not hardcoded weights.
        
        Args:
            tokens: Ordered list of tokens from gdino_tokens (index 0 = highest priority)
            max_tokens: Maximum number of tokens to consider
            
        Returns:
            List of tokens with pure hierarchical priority weighting
        """
        if not tokens:
            return []
        
        # Noise terms to filter out (keeping minimal set)
        noise_terms = {
            'de', 'en', 'com', 'por', 'para', 'con', 'las', 'los', 'el', 'la', 'w', 'r', 'l', 
            'buy', 'best', 'online', 'youtube', 'tiktok', 'facebook', 'mercadolibre', 'wallapop', 
            'amazon', 'ebay', 'price', 'precio', 'estado', 'condition', 'segunda', 
            'mano', 'bundle', 'set', 'kit', 'box', 'caja', 'メルカリ', 'ラクマ', 'yahoo', 
            'オークション', 'フリマ', 'eur', 'usd', 'shipping', 'free'
        }
        
        # Hardware terms for identification (NO WEIGHTING - equal treatment)
        hardware_terms = {
            'nintendo', 'playstation', 'xbox', 'switch', 'controller', 'console',
            'ds', 'dsi', 'psp', 'vita', 'gamecube', 'wii', 'joy', 'con', 'joycon',
            'ps1', 'ps2', 'ps3', 'ps4', 'ps5', '3ds', 'gba', 'n64', 'snes',
            'scph', 'oled', 'pro', 'slim', 'lite', 'memory', 'card', 'handheld',
            'dualshock', 'sixaxis', 'dualsense', 'gameboy', 'saturn', 'genesis',
            'original', 'model', 'ntr', 'dol', 'charger', 'tested', 'working'
        }
        
        prioritized_tokens = []
        
        # Only process top tokens for efficiency (default 10)
        for idx, token in enumerate(tokens[:max_tokens]):
            token_clean = token.strip().lower()
            
            # Skip empty or very short tokens
            if len(token_clean) < 2:
                continue
                
            # Skip noise terms - just ignore, no penalties
            if token_clean in noise_terms:
                continue
            
            # Use CategoryDetector for intelligent filtering (if available)
            if (self.category_detector and self.category_detector.is_trained):
                # Quick category check for this single token
                single_token_result = self.category_detector.detect_token_categories([token], top_k=1)
                # Skip if clearly non-gaming with high confidence
                if not single_token_result['is_gaming'] and single_token_result['confidence'] > 0.7:
                    continue
            else:
                # Fallback: basic non-gaming detection
                non_gaming_indicators = {
                    'mouse', 'mice', 'rat', 'bike', 'bicycle', 'car', 'auto', 'vehicle',
                    'clock', 'watch', 'time', 'alarm', 'phone', 'mobile', 'cell'
                }
                if any(non_gaming in token_clean for non_gaming in non_gaming_indicators):
                    continue  # Simply skip non-gaming tokens
            
            # Position-based exponential decay weighting
            # Earlier positions get exponentially higher priority
            decay_factor = 0.8  # Configurable decay rate
            base_weight = 1.0
            hierarchy_weight = base_weight * (decay_factor ** idx)  # 1.0, 0.8, 0.64, 0.51, 0.41...
            
            # Determine if token is hardware-related using learned embeddings
            if (self.category_detector and self.category_detector.is_trained):
                single_token_result = self.category_detector.detect_token_categories([token], top_k=1)
                is_hardware_related = single_token_result['is_gaming']
            else:
                # Fallback: use basic hardware terms
                is_hardware_related = any(hw in token_clean or token_clean in hw for hw in hardware_terms)
            
            prioritized_tokens.append({
                'token': token,
                'token_clean': token_clean,
                'position_index': idx,
                'hierarchy_weight': hierarchy_weight,
                'is_hardware_related': is_hardware_related,
                # Priority is PURELY based on position in gdino_tokens
                'priority_score': 1000 - idx  # Simple: earlier = higher score
            })
        
        # Sort by position index to maintain original hierarchy (DO NOT sort by weight)
        prioritized_tokens.sort(key=lambda x: x['position_index'])
        
        return prioritized_tokens

    # LEGACY CODE - COMMENTED OUT FOR REFERENCE
    # def apply_dynamic_token_weighting(self, tokens: List[str], max_tokens: int = 10, 
    #                                  decay_factor: float = 0.1, min_weight: float = 0.1) -> List[Dict[str, Any]]:
    #     """
    #     LEGACY: Old approach with hardcoded hardware keyword weighting.
    #     Replaced with pure hierarchical approach based on gdino_tokens position.
    #     """
    #     pass

    def search_similar_items(self, tokens: List[str], k: int = 5, use_hierarchy: bool = True, 
                           weighting_config: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar items based on token array using hierarchical priority system.
        
        Args:
            tokens: Ordered list of detection tokens (index 0 = highest priority)
            k: Number of similar items to return
            use_hierarchy: Enable hierarchical token priority system
            weighting_config: Configuration for hierarchy algorithm
            
        Returns:
            List of similar items with similarity scores
        """
        if not tokens:
            return []
        
        # Use exact matches first, then fall back to semantic search
        exact_matches = self.find_exact_matches(tokens)
        if exact_matches:
            # Convert exact matches to similarity results format
            results = []
            for match in exact_matches[:k]:
                item_data = match['item'].copy()
                item_data['similarity_score'] = 1.0  # Exact matches get perfect score
                item_data['match_type'] = match['match_type']
                item_data['hierarchy_score'] = match['hierarchy_score']
                results.append(item_data)
            return results
        
        # Fallback to semantic search with hierarchical token priority
        if weighting_config is None:
            weighting_config = {
                'max_tokens': 15,  # Increased to consider more tokens
                'use_hierarchy': True
            }
        
        try:
            if use_hierarchy and weighting_config.get('use_hierarchy', True):
                # Apply hierarchical token priority based on gdino_tokens position
                prioritized_tokens = self.apply_hierarchical_token_priority(
                    tokens, 
                    max_tokens=weighting_config['max_tokens']
                )
                
                if not prioritized_tokens:
                    logger.debug("No valid tokens after hierarchical processing")
                    return []
                
                # Create query from prioritized tokens (maintain gdino_tokens order)
                query_tokens = [pt['token'] for pt in prioritized_tokens[:15]]
                
                logger.debug(f"Hierarchical priority applied: {len(prioritized_tokens)} valid tokens")
                top_tokens_str = ', '.join([f"{pt['token']}(pos:{pt['position_index']})" for pt in prioritized_tokens[:5]])
                logger.debug(f"Top 5 priority tokens: {top_tokens_str}")
                
            else:
                # Simple fallback filtering
                filtered_tokens = []
                for token in tokens[:20]:
                    token_clean = token.strip().lower()
                    if len(token_clean) >= 2:
                        filtered_tokens.append(token)
                
                query_tokens = filtered_tokens
            
            if not query_tokens:
                return []
            
            # Create position-weighted embeddings respecting gdino_tokens exact order
            if use_hierarchy and len(prioritized_tokens) > 0:
                # Build weighted query based on exact position weights
                weighted_query_parts = []
                
                for pt in prioritized_tokens[:10]:  # Use top 10 prioritized tokens
                    token = pt['token']
                    weight = pt['hierarchy_weight']
                    
                    # Repeat tokens based on their exponential decay weight
                    # Higher weights = more repetitions in query
                    repetitions = max(1, int(weight * 5))  # Scale weight to repetition count
                    weighted_query_parts.extend([token] * repetitions)
                
                query_text = ' '.join(weighted_query_parts)
            else:
                # Fallback: simple join
                query_text = ' '.join(query_tokens)
            
            # Add E5 query prefix for better retrieval performance
            query_text = f"query: {query_text}"
            
            # Encode query
            query_embedding = self.model.encode([query_text])
            faiss.normalize_L2(query_embedding.astype('float32'))
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            # Format results with hierarchy information
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score > 0.0:  # Valid result with positive similarity
                    item_meta = self.metadata[idx].copy()
                    item_meta['similarity_score'] = float(score)
                    item_meta['match_type'] = 'semantic_similarity'
                    results.append(item_meta)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for tokens: {e}")
            return []

    def search_itemtypes(self, tokens: List[str], k: int = 5, use_hierarchy: bool = True,
                       weighting_config: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar items in the itemtypes database using hierarchical priority system.
        
        Args:
            tokens: Ordered list of detection tokens (index 0 = highest priority)
            k: Number of similar items to return
            use_hierarchy: Enable hierarchical token priority system
            weighting_config: Configuration for hierarchy algorithm
            
        Returns:
            List of similar items with similarity scores
        """
        if not tokens or self.itemtypes_index is None:
            return []
        
        # Use hierarchical token priority for itemtypes search
        if weighting_config is None:
            weighting_config = {
                'max_tokens': 10,
                'use_hierarchy': True
            }
        
        try:
            if use_hierarchy and weighting_config.get('use_hierarchy', True):
                # Apply hierarchical token priority for itemtypes
                prioritized_tokens = self.apply_hierarchical_token_priority(
                    tokens, 
                    max_tokens=weighting_config['max_tokens']
                )
                
                if not prioritized_tokens:
                    logger.debug("Itemtypes: No valid tokens after hierarchical processing")
                    return []
                
                # Create query from prioritized tokens (maintain gdino_tokens order)
                query_tokens = [pt['token'] for pt in prioritized_tokens[:15]]
                
                logger.debug(f"Itemtypes hierarchical priority: {len(prioritized_tokens)} valid tokens")
                
            else:
                # Simple fallback filtering
                filtered_tokens = []
                for token in tokens[:20]:
                    token_clean = token.strip().lower()
                    if len(token_clean) >= 2:
                        filtered_tokens.append(token)
                
                query_tokens = filtered_tokens
            
            if not query_tokens:
                return []
            
            # Create position-weighted embeddings respecting gdino_tokens exact order
            if use_hierarchy and len(prioritized_tokens) > 0:
                # Build weighted query based on exact position weights
                weighted_query_parts = []
                
                for pt in prioritized_tokens[:10]:  # Use top 10 prioritized tokens
                    token = pt['token']
                    weight = pt['hierarchy_weight']
                    
                    # Repeat tokens based on their exponential decay weight
                    # Higher weights = more repetitions in query
                    repetitions = max(1, int(weight * 5))  # Scale weight to repetition count
                    weighted_query_parts.extend([token] * repetitions)
                
                query_text = ' '.join(weighted_query_parts)
            else:
                # Fallback: simple join
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

    def find_itemtypes_match(self, tokens: List[str], min_similarity: float = 0.3, use_hierarchy: bool = True) -> Optional[Dict[str, Any]]:
        """
        Find the best itemtypes match for given tokens using hierarchical token priority.
        
        Args:
            tokens: Detection tokens from gdino_tokens (ordered by priority)
            min_similarity: Minimum similarity threshold
            use_hierarchy: Enable hierarchical token priority system
            
        Returns:
            Best itemtypes match or None
        """
        if not self.itemtypes_index:
            return None
        
        # First try exact matches
        exact_matches = self.find_exact_matches(tokens)
        itemtype_exact_matches = [m for m in exact_matches if m['source'] == 'itemtypes']
        
        if itemtype_exact_matches:
            # Return the highest priority exact match
            best_exact = itemtype_exact_matches[0]
            result = best_exact['item'].copy()
            result['similarity_score'] = 1.0  # Exact match
            result['match_type'] = best_exact['match_type']
            result['hierarchy_analysis'] = {
                'match_token': best_exact.get('token', best_exact.get('phrase', 'unknown')),
                'token_position': best_exact['token_index'],
                'match_field': best_exact['matched_field'],
                'is_exact_match': True
            }
            return result
            
        # Fallback to semantic similarity
        itemtype_results = self.search_itemtypes(tokens, k=3, use_hierarchy=use_hierarchy)
        if itemtype_results and itemtype_results[0]['similarity_score'] >= min_similarity:
            # Add hierarchical analysis metadata
            if use_hierarchy:
                prioritized_tokens = self.apply_hierarchical_token_priority(tokens)
                if prioritized_tokens:
                    itemtype_results[0]['hierarchy_analysis'] = {
                        'top_priority_tokens': [pt['token'] for pt in prioritized_tokens[:5]],
                        'total_valid_tokens': len(prioritized_tokens),
                        'highest_priority_score': prioritized_tokens[0]['priority_score'] if prioritized_tokens else 0.0,
                        'has_hardware_terms': any(pt['is_hardware_related'] for pt in prioritized_tokens),
                        'is_exact_match': False
                    }
            return itemtype_results[0]
        return None

    def assess_token_quality(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Assess the quality of tokens for classification purposes using learned embeddings.
        
        Args:
            tokens: List of tokens to assess
            
        Returns:
            Dictionary with quality metrics and filtered tokens
        """
        if not tokens:
            return {'quality_score': 0.0, 'valid_tokens': [], 'has_hardware_terms': False, 'gaming_relevance': 0.0}
        
        # Use CategoryDetector for learned classification
        if self.category_detector and self.category_detector.is_trained:
            category_result = self.category_detector.detect_token_categories(tokens, top_k=3)
            
            # Enhanced quality assessment with "no match" logic for very poor tokens
            quality_score = category_result['confidence']
            gaming_relevance = category_result['confidence'] if category_result['is_gaming'] else 0.0
            
            # Check for very poor token quality that should result in "no match"
            very_poor_quality = self._assess_token_degradation(tokens[:20])
            
            # If tokens are of very poor quality, force "no match"
            if very_poor_quality['is_very_poor']:
                logger.debug(f"Very poor token quality detected: {very_poor_quality['reason']}")
                quality_score = 0.0
                gaming_relevance = 0.0
                category_result['is_gaming'] = False
            
            return {
                'quality_score': quality_score,
                'valid_tokens': category_result['valid_tokens'],
                'hardware_tokens': category_result['valid_tokens'] if category_result['is_gaming'] else [],
                'non_gaming_tokens': [],  # Handled internally by CategoryDetector
                'has_hardware_terms': category_result['is_gaming'],
                'hardware_ratio': 1.0 if category_result['is_gaming'] else 0.0,
                'valid_ratio': 1.0,
                'non_gaming_ratio': 0.0,
                'gaming_relevance': gaming_relevance,
                'total_tokens': len(tokens),
                'detected_categories': category_result['categories'],
                'token_quality_check': very_poor_quality
            }
        
        # Fallback to basic filtering when CategoryDetector not available
        else:
            logger.warning("CategoryDetector not trained, using basic token filtering")
            
            # Invalid token patterns (garbled text, special characters, etc.)
            invalid_patterns = [
                r'^[■▪▫□◯○●◆◇△▲▼◄►♠♥♦♣]+$',  # Special characters only
                r'^[\d\s\W]+$',  # Only numbers/spaces/special chars
                r'^.{1}$',  # Single character tokens
                r'^[xX]+$',  # Just x's
            ]
            
            # Common irrelevant terms
            irrelevant_terms = {
                'yahoo', 'ebay', 'amazon', 'mercado', 'libre', 'wallapop', 'facebook',
                'tiktok', 'youtube', 'instagram', 'twitter', 'com', 'www', 'http',
                'price', 'precio', 'estado', 'condition', 'shipping', 'free', 'best',
                'buy', 'sale', 'nuevo', 'usado', 'second', 'hand', 'mano', 'de',
                'en', 'el', 'la', 'los', 'las', 'por', 'para', 'con', 'sin'
            }
            
            valid_tokens = []
            quality_score = 0.0
            
            import re
            
            # Only process top 10 tokens
            top_tokens = tokens[:20]
            
            for token in top_tokens:
                token_clean = token.strip().lower()
                
                # Skip empty or very short tokens
                if len(token_clean) < 2:
                    continue
                
                # Check for invalid patterns
                is_invalid = any(re.match(pattern, token, re.IGNORECASE) for pattern in invalid_patterns)
                if is_invalid:
                    continue
                
                # Skip irrelevant terms
                if token_clean in irrelevant_terms:
                    continue
                
                # Valid token - add to list
                valid_tokens.append(token)
                quality_score += 0.5
        
            # Calculate basic quality metrics
            top_token_count = len(top_tokens)
            valid_ratio = len(valid_tokens) / top_token_count if top_token_count else 0
            
            # Simple quality score
            final_quality = min(1.0, max(0.0, quality_score / len(valid_tokens))) if valid_tokens else 0.0
            
            # Hardware/gaming term detection for fallback
            hardware_terms = {
                'nintendo', 'playstation', 'xbox', 'switch', 'controller', 'console',
                'ds', 'dsi', 'psp', 'vita', 'gamecube', 'wii', 'joy', 'con', 'joycon',
                'ps1', 'ps2', 'ps3', 'ps4', 'ps5', '3ds', 'gba', 'n64', 'snes',
                'scph', 'oled', 'pro', 'slim', 'lite', 'memory', 'card', 'handheld',
                'dualshock', 'sixaxis', 'dualsense', 'gameboy', 'saturn', 'genesis',
                'original', 'model', 'ntr', 'dol', 'charger', 'tested', 'working'
            }
            
            hardware_matches = []
            for token in valid_tokens:
                token_lower = token.lower().strip()
                if token_lower in hardware_terms:
                    hardware_matches.append(token)
                else:
                    # Check if any hardware term is in the token
                    if any(hw_term in token_lower for hw_term in hardware_terms):
                        hardware_matches.append(token)
            
            hardware_ratio = len(hardware_matches) / len(valid_tokens) if valid_tokens else 0.0
            gaming_relevance = hardware_ratio  # Use hardware ratio as gaming relevance
            
            return {
                'quality_score': final_quality,
                'valid_tokens': valid_tokens,
                'hardware_tokens': hardware_matches,
                'non_gaming_tokens': [],
                'has_hardware_terms': len(hardware_matches) > 0,
                'hardware_ratio': hardware_ratio,
                'valid_ratio': valid_ratio,
                'non_gaming_ratio': 0.0,
                'gaming_relevance': gaming_relevance,
                'total_tokens': len(tokens),
                'valid_token_count': len(valid_tokens)
            }
    
    def _assess_token_degradation(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Assess if tokens are of very poor quality that should result in "no match".
        
        Args:
            tokens: List of tokens to assess (top 20)
            
        Returns:
            Dict with degradation assessment
        """
        if not tokens:
            return {'is_very_poor': True, 'reason': 'No tokens provided'}
        
        import re
        
        # Count various quality issues
        very_short_tokens = sum(1 for token in tokens if len(token.strip()) <= 1)
        gibberish_tokens = sum(1 for token in tokens 
                             if re.match(r'^[^a-zA-Z0-9\s]{3,}$', token) or 
                                len(re.sub(r'[a-zA-Z0-9\s]', '', token)) > len(token) * 0.7)
        numeric_only_tokens = sum(1 for token in tokens if token.strip().isdigit())
        
        # Language/marketplace noise
        marketplace_noise = {
            'ebay', 'amazon', 'mercado', 'libre', 'wallapop', 'facebook',
            'youtube', 'instagram', 'tiktok', 'precio', 'price', 'shipping',
            'buy', 'sell', 'new', 'used', 'condition', 'estado', 'usado'
        }
        noise_tokens = sum(1 for token in tokens 
                          if token.lower().strip() in marketplace_noise)
        
        # Single character tokens
        single_char_tokens = sum(1 for token in tokens 
                               if len(token.strip()) == 1 and token.isalpha())
        
        # URL/HTML fragments
        url_fragments = sum(1 for token in tokens 
                          if any(fragment in token.lower() 
                               for fragment in ['http', 'www', '.com', '.html', 'href']))
        
        total_tokens = len(tokens)
        
        # Calculate degradation ratios
        poor_quality_ratio = (
            very_short_tokens + gibberish_tokens + single_char_tokens + url_fragments
        ) / total_tokens
        
        noise_ratio = noise_tokens / total_tokens
        numeric_ratio = numeric_only_tokens / total_tokens
        
        # Determine if very poor quality
        is_very_poor = False
        reason = "Good quality"
        
        if poor_quality_ratio > 0.6:  # More than 60% poor quality tokens
            is_very_poor = True
            reason = f"Too many poor quality tokens ({poor_quality_ratio:.1%})"
        elif noise_ratio > 0.5:  # More than 50% marketplace noise
            is_very_poor = True
            reason = f"Too much marketplace noise ({noise_ratio:.1%})"
        elif numeric_ratio > 0.7:  # More than 70% numbers
            is_very_poor = True
            reason = f"Too many numeric tokens ({numeric_ratio:.1%})"
        elif total_tokens < 3:  # Too few tokens to make meaningful assessment
            is_very_poor = True
            reason = "Too few meaningful tokens"
        elif very_short_tokens == total_tokens:  # All tokens are too short
            is_very_poor = True
            reason = "All tokens are too short"
        
        return {
            'is_very_poor': is_very_poor,
            'reason': reason,
            'poor_quality_ratio': poor_quality_ratio,
            'noise_ratio': noise_ratio,
            'numeric_ratio': numeric_ratio,
            'quality_breakdown': {
                'very_short': very_short_tokens,
                'gibberish': gibberish_tokens,
                'numeric_only': numeric_only_tokens,
                'noise': noise_tokens,
                'single_char': single_char_tokens,
                'url_fragments': url_fragments,
                'total': total_tokens
            }
        }

    def detect_brand_from_tokens(self, tokens: List[str]) -> Optional[str]:
        """
        Detect the primary brand from tokens for consistency validation.
        
        Args:
            tokens: List of tokens to analyze
            
        Returns:
            Detected brand string or None
        """
        # Brand detection mapping with priority (more specific terms first)
        brand_indicators = {
            'playstation': {'brand': 'Sony', 'aliases': ['sony', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'playstation', 'プレイステーション', 'dualshock', 'sixaxis', 'dualsense']},
            'nintendo': {'brand': 'Nintendo', 'aliases': ['nintendo', '任天堂', 'ds', 'dsi', '3ds', 'gba', 'n64', 'snes', 'wii', 'switch', 'gamecube', 'gameboy', 'joycon']},
            'xbox': {'brand': 'Microsoft', 'aliases': ['xbox', 'microsoft']},
            'sega': {'brand': 'Sega', 'aliases': ['sega', 'saturn', 'genesis', 'dreamcast']},
            'konami': {'brand': 'Konami', 'aliases': ['konami', 'コナミ']}
        }
        
        # Count brand indicators in top tokens (position-weighted)
        brand_scores = {}
        
        for idx, token in enumerate(tokens[:10]):  # Check top 10 tokens
            token_clean = token.lower().strip()
            
            # Position weight (earlier tokens have higher impact)
            position_weight = 1.0 * (0.8 ** idx)
            
            for brand_key, brand_info in brand_indicators.items():
                if any(alias in token_clean or token_clean in alias for alias in brand_info['aliases']):
                    brand = brand_info['brand']
                    if brand not in brand_scores:
                        brand_scores[brand] = 0.0
                    brand_scores[brand] += position_weight
        
        # Return brand with highest score, if above threshold
        if brand_scores:
            best_brand = max(brand_scores, key=brand_scores.get)
            if brand_scores[best_brand] >= 0.3:  # Minimum confidence threshold
                return best_brand
        
        return None

    def validate_brand_consistency(self, tokens: List[str], match_result: Dict[str, Any]) -> bool:
        """
        Validate that the match result is consistent with the detected brand from tokens.
        
        Args:
            tokens: Original detection tokens
            match_result: Proposed match result
            
        Returns:
            True if brands are consistent, False if there's a mismatch
        """
        detected_brand = self.detect_brand_from_tokens(tokens)
        
        if not detected_brand:
            return True  # No strong brand signal, allow match
        
        # Get brand from match result
        match_brand = None
        if match_result.get('brand'):
            match_brand = match_result['brand']
        elif 'nintendo' in match_result.get('name', '').lower():
            match_brand = 'Nintendo'
        elif any(term in match_result.get('name', '').lower() for term in ['playstation', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'sony']):
            match_brand = 'Sony'
        elif 'xbox' in match_result.get('name', '').lower():
            match_brand = 'Microsoft'
        elif 'sega' in match_result.get('name', '').lower():
            match_brand = 'Sega'
        elif 'konami' in match_result.get('name', '').lower():
            match_brand = 'Konami'
        
        # Check for brand consistency
        if match_brand and detected_brand != match_brand:
            logger.debug(f"Brand mismatch: detected '{detected_brand}' from tokens, but match is '{match_brand}' - rejecting match")
            return False
        
        return True

    def _get_correct_vector_id(self, item_data: Dict[str, Any], source: str) -> Optional[str]:
        """
        Get the correct vector database ID for an item, ensuring proper mapping from itemtypes to vector database.
        
        Args:
            item_data: Item data from itemtypes or vector database
            source: Source database ('itemtypes' or 'vector_database')
            
        Returns:
            Correct vector database ID or None if not found/irrelevant
        """
        # If already from vector database, validate and return ID
        if source == 'vector_database':
            item_id = item_data.get('id')
            if item_id and item_id in self.items_data:
                return item_id
        
        # For itemtypes, find corresponding vector database ID
        if source == 'itemtypes':
            item_name = item_data.get('name', '').lower().strip()
            item_category = item_data.get('category', '').lower()
            item_brand = item_data.get('brand', '').lower()
            
            if not item_name:
                return None
            
            # Strategy 1: Search by exact name match in vector database
            for vector_id, vector_item in self.items_data.items():
                vector_name = vector_item.get('name', '').lower().strip()
                vector_category = vector_item.get('category', '').lower()
                vector_brand = vector_item.get('brand', '').lower()
                
                # Exact name match with category/brand validation
                if vector_name == item_name:
                    # Additional validation for accuracy
                    category_match = not item_category or not vector_category or item_category == vector_category
                    brand_match = not item_brand or not vector_brand or item_brand == vector_brand
                    
                    if category_match and brand_match:
                        # Check if this item is relevant (has good hardware relevance score)
                        if self._is_vector_item_relevant(vector_id):
                            return vector_id
            
            # Strategy 2: Fuzzy name matching with high threshold
            for vector_id, vector_item in self.items_data.items():
                vector_name = vector_item.get('name', '').lower().strip()
                
                # Calculate similarity
                similarity = self._calculate_name_similarity(item_name, vector_name)
                
                if similarity > 0.9:  # Very high threshold for fuzzy matching
                    if self._is_vector_item_relevant(vector_id):
                        return vector_id
        
        return None
    
    def _is_vector_item_relevant(self, vector_id: str) -> bool:
        """
        Check if a vector database item is relevant (not a low-quality or irrelevant item).
        
        Args:
            vector_id: Vector database item ID
            
        Returns:
            True if relevant, False if should be filtered out
        """
        if not self.item_lookup or vector_id not in self.item_lookup:
            return True  # Default to relevant if no lookup data
        
        lookup_data = self.item_lookup[vector_id]
        
        # Check hardware relevance score
        hardware_relevance = lookup_data.get('hardware_relevance_score', 1.0)
        if hardware_relevance < 0.1:
            return False
        
        # Check for obviously irrelevant categories
        category = lookup_data.get('category', '').lower()
        irrelevant_categories = {
            'labels', 'shipping', 'office supplies', 'furniture', 
            'clothing', 'food', 'automotive', 'books', 'music',
            'home & garden', 'health & beauty', 'tools'
        }
        
        if any(irrelevant in category for irrelevant in irrelevant_categories):
            return False
        
        # Check name for obvious non-gaming terms
        name = lookup_data.get('name', '') or ''
        name = name.lower() if name else ''
        irrelevant_terms = {
            'shipping label', 'mouse pad', 'office chair', 'bicycle', 
            'clock', 'watch', 'phone case', 'car', 'book', 'cd music',
            'sticker', 'poster', 'keychain'
        }
        
        if name and any(term in name for term in irrelevant_terms):
            return False
        
        return True
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two item names using token overlap."""
        if not name1 or not name2:
            return 0.0
        
        # Simple token-based similarity
        tokens1 = set(name1.lower().split())
        tokens2 = set(name2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0

    def get_best_classification(self, tokens: List[str], min_similarity: float = 0.5, 
                              use_hierarchy: bool = True, 
                              hierarchy_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get the best classification for tokens using industry-level hierarchical priority system.
        Prioritizes exact matches over semantic similarity based on gdino_tokens position.
        
        Args:
            tokens: Ordered detection tokens from gdino_tokens (index 0 = highest priority)
            min_similarity: Minimum similarity threshold for valid classification
            use_hierarchy: Enable hierarchical token priority system
            hierarchy_config: Configuration for hierarchy system
            
        Returns:
            Dict with complete metadata or None for "No Match" cases
        """
        if not tokens:
            logger.debug("No tokens provided - returning None")
            return None
        
        # Configure hierarchical system parameters
        if hierarchy_config is None:
            hierarchy_config = {
                'max_tokens': 15,
                'prioritize_exact_matches': True,
                'compound_phrase_detection': True
            }
        
        # INDUSTRY-LEVEL APPROACH: Check exact matches FIRST
        if hierarchy_config.get('prioritize_exact_matches', True):
            exact_matches = self.find_exact_matches(tokens)
            if exact_matches:
                # Found exact match! This takes absolute priority
                best_exact = exact_matches[0]  # Already sorted by hierarchy score
                matched_text = best_exact.get('token', best_exact.get('phrase', 'unknown'))
                logger.debug(f"EXACT MATCH FOUND: {matched_text} at position {best_exact['token_index']} -> {best_exact['item']['name']}")
                
                # CRITICAL FIX: Always use proper vector database ID
                correct_vector_id = self._get_correct_vector_id(best_exact['item'], best_exact['source'])
                if not correct_vector_id:
                    # If no proper vector ID found, mark as irrelevant
                    logger.debug(f"No valid vector database ID found for {best_exact['item']['name']} - marking as No Match")
                    return None
                
                # Get the correct item data from vector database
                vector_item_data = self.items_data.get(correct_vector_id, {})
                vector_lookup_data = self.item_lookup.get(correct_vector_id, {}) if self.item_lookup else {}
                
                result = {
                    'id': correct_vector_id,  # ✅ ALWAYS use correct vector database ID
                    'reference_name': vector_item_data.get('name', best_exact['item'].get('name')),
                    'itemtypes_name': best_exact['item'].get('name') if best_exact['source'] == 'itemtypes' else None,
                    'category': vector_item_data.get('category', best_exact['item']['category']),
                    'model': vector_item_data.get('model', best_exact['item'].get('model', '')),
                    'brand': vector_item_data.get('brand', best_exact['item'].get('brand', '')),
                    'similarity_score': 1.0,  # Exact match = perfect score
                    'source': 'vector_database',  # ✅ Always mark as vector database source
                    'contextual_text': vector_lookup_data.get('contextual_text', best_exact['item'].get('contextual_text', '')),
                    'vector_index': vector_lookup_data.get('vector_index'),
                    'embedding_norm': vector_lookup_data.get('embedding_norm', 1.0),
                    'legacyId': vector_item_data.get('legacyId'),
                    'vector_reference_id': correct_vector_id,  # ✅ Consistent reference
                    'vector_reference_name': vector_item_data.get('name', best_exact['item'].get('name')),
                    'match_type': best_exact['match_type'],
                    'match_details': {
                        'matched_token': best_exact.get('token', best_exact.get('phrase', 'unknown')),
                        'token_position': best_exact['token_index'],
                        'matched_field': best_exact['matched_field'],
                        'hierarchy_score': best_exact['hierarchy_score'],
                        'is_exact_match': True,
                        'is_compound_match': 'compound' in best_exact['match_type'],
                        'original_itemtype_id': best_exact['item']['id'] if best_exact['source'] == 'itemtypes' else None
                    }
                }
                
                # All data is already correct from vector database - no need for enrichment
                
                logger.debug(f"Exact match classification: {result.get('itemtypes_name') or result.get('reference_name')}")
                return result
        
        # No exact matches found, assess token quality for semantic search
        token_quality = self.assess_token_quality(tokens)
        
        # Apply basic quality filter
        quality_threshold = 0.15  # Slightly lower threshold since we have hierarchy
        if token_quality['quality_score'] < quality_threshold:
            logger.debug(f"Token quality too low: {token_quality['quality_score']:.3f} < {quality_threshold} - No Match")
            return None
        
        # Apply hierarchical token priority for semantic search
        if use_hierarchy:
            prioritized_tokens = self.apply_hierarchical_token_priority(
                tokens, max_tokens=hierarchy_config['max_tokens']
            )
            
            if not prioritized_tokens:
                logger.debug("No valid prioritized tokens - No Match")
                return None
                
            logger.debug(f"Hierarchical priority: {len(prioritized_tokens)} valid tokens from {len(tokens)} total")
        
        # Search both databases using hierarchical priority
        itemtype_match = self.find_itemtypes_match(
            tokens, min_similarity, use_hierarchy=use_hierarchy
        )
        
        similar_items = self.search_similar_items(
            tokens, k=3, use_hierarchy=use_hierarchy, 
            weighting_config={'max_tokens': 15, 'use_hierarchy': True}
        )
        vector_match = None
        if similar_items and similar_items[0]['similarity_score'] >= min_similarity:
            vector_match = similar_items[0]
        
        # Hierarchical approach: prioritize matches with earlier token positions
        adjusted_min_similarity = min_similarity
        if token_quality['has_hardware_terms'] and prioritized_tokens:
            # For hardware-relevant tokens, slightly lower threshold
            hardware_token_count = sum(1 for pt in prioritized_tokens if pt['is_hardware_related'])
            if hardware_token_count > 0:
                adjusted_min_similarity = min_similarity * 0.95
                logger.debug(f"Hardware terms detected, threshold adjusted to {adjusted_min_similarity:.3f}")
        
        # Re-evaluate matches with adjusted threshold
        if itemtype_match and itemtype_match['similarity_score'] < adjusted_min_similarity:
            logger.debug(f"Itemtype match below threshold: {itemtype_match['similarity_score']:.3f} < {adjusted_min_similarity:.3f}")
            itemtype_match = None
        if vector_match and vector_match['similarity_score'] < adjusted_min_similarity:
            logger.debug(f"Vector match below threshold: {vector_match['similarity_score']:.3f} < {adjusted_min_similarity:.3f}")
            vector_match = None
        
        # If no semantic matches meet threshold, return None
        if not itemtype_match and not vector_match:
            logger.debug(f"No semantic matches above threshold {adjusted_min_similarity:.3f} - No Match")
            return None
        
        # Apply brand consistency validation before accepting matches
        if itemtype_match and not self.validate_brand_consistency(tokens, itemtype_match):
            logger.debug("Itemtype match rejected due to brand inconsistency")
            itemtype_match = None
        
        if vector_match and not self.validate_brand_consistency(tokens, vector_match):
            logger.debug("Vector match rejected due to brand inconsistency")  
            vector_match = None
        
        # Apply token existence validation to prevent over-specification
        if itemtype_match and not self._validate_token_existence(tokens, itemtype_match.get('name', '')):
            logger.debug("Itemtype match rejected due to non-existent terms in tokens")
            itemtype_match = None
            
        if vector_match and not self._validate_token_existence(tokens, vector_match.get('name', '')):
            logger.debug("Vector match rejected due to non-existent terms in tokens")
            vector_match = None
        
        # If no matches pass brand validation, return None
        if not itemtype_match and not vector_match:
            logger.debug("No matches passed brand consistency validation - No Match")
            return None
        
        # ENFORCE: Always use items.json (vector database) for final ID - no exceptions
        if not vector_match:
            # If no vector match, try to find one with lower threshold for items.json ID
            fallback_vector_results = self.search_similar_items(
                tokens, k=5, use_hierarchy=use_hierarchy, 
                weighting_config={'max_tokens': 15, 'use_hierarchy': True}
            )
            if fallback_vector_results and fallback_vector_results[0]['similarity_score'] >= 0.3:
                vector_match = fallback_vector_results[0]
                logger.debug(f"Found fallback vector match for items.json ID: {vector_match['name']} (similarity: {vector_match['similarity_score']:.3f})")
            else:
                logger.debug("No items.json match found - returning None (accuracy over guessing)")
                return None  # Must have items.json ID - no guessing
        
        # Validate vector match has terms that exist in tokens
        if not self._validate_token_existence(tokens, vector_match.get('name', '')):
            logger.debug(f"Vector match '{vector_match['name']}' contains non-existent terms - returning None for accuracy")
            return None
        elif itemtype_match:
            # Even if only itemtypes matches, try to find vector database equivalent for proper ID
            # Search with lower threshold to find vector database match
            fallback_vector_results = self.search_similar_items(
                tokens, k=5, use_hierarchy=use_hierarchy, 
                weighting_config={'max_tokens': 15, 'use_hierarchy': True}
            )
            if fallback_vector_results and fallback_vector_results[0]['similarity_score'] >= min_similarity * 0.7:
                # Found a reasonable vector match, use it for proper ID
                primary_source = 'vector_database'
                vector_match = fallback_vector_results[0]
            else:
                primary_source = 'itemtypes'
        elif vector_match:
            primary_source = 'vector_database'
        
        # Build result - ALWAYS use vector database (items.json) ID 
        original_id = vector_match['id']
        original_metadata = self.item_lookup.get(original_id, {}) if self.item_lookup else {}
        
        result = {
            'id': original_id,  # ALWAYS from items.json (vector database) - no exceptions
            'reference_name': vector_match['name'],  # Name from items.json
            'itemtypes_name': None,  # Enhanced naming from itemtypes (if any)
            'category': vector_match['category'],
            'model': vector_match.get('model', ''),
            'brand': '',  # Will be enhanced from itemtypes if available and validated
            'similarity_score': vector_match['similarity_score'],
            'source': 'vector_database',  # Always vector_database for items.json IDs
            'contextual_text': original_metadata.get('contextual_text', ''),
            'vector_index': original_metadata.get('vector_index', None),
            'embedding_norm': original_metadata.get('embedding_norm', None),
            'legacyId': original_metadata.get('legacyId', None)
        }
        
        # Use itemtypes only for enhanced naming (and only if terms exist in tokens)
        if itemtype_match and self._validate_token_existence(tokens, itemtype_match.get('name', '')):
            result['itemtypes_name'] = itemtype_match['name']
            result['itemtypes_similarity'] = itemtype_match['similarity_score']
            # Update brand if itemtypes has brand info
            if itemtype_match.get('brand'):
                result['brand'] = itemtype_match['brand']
            # Primary match is vector database - validate and use proper ID
            correct_vector_id = self._get_correct_vector_id(vector_match, 'vector_database')
            if not correct_vector_id:
                # If no proper vector ID found, mark as irrelevant
                logger.debug(f"No valid vector database ID found for vector match {vector_match['name']} - marking as No Match")
                return None
            
            # Get the correct item data
            vector_item_data = self.items_data.get(correct_vector_id, {})
            vector_lookup_data = self.item_lookup.get(correct_vector_id, {}) if self.item_lookup else {}
            
            result = {
                'id': correct_vector_id,  # ✅ Validated vector database ID
                'reference_name': vector_item_data.get('name', vector_match['name']),
                'itemtypes_name': None,  # Will be set below if itemtype_match exists
                'category': vector_item_data.get('category', vector_match['category']),
                'model': vector_item_data.get('model', vector_match.get('model', '')),
                'brand': vector_item_data.get('brand', ''),
                'similarity_score': vector_match['similarity_score'],
                'source': 'vector_database',
                'contextual_text': vector_lookup_data.get('contextual_text', ''),
                'vector_index': vector_lookup_data.get('vector_index'),
                'embedding_norm': vector_lookup_data.get('embedding_norm', 1.0),
                'legacyId': vector_item_data.get('legacyId'),
                'vector_reference_id': correct_vector_id,  # ✅ Consistent reference
                'vector_reference_name': vector_item_data.get('name', vector_match['name'])
            }
            
            # Try to find corresponding itemtypes match for enhanced naming
            if itemtype_match:
                result['itemtypes_name'] = itemtype_match['name']
                result['itemtypes_similarity'] = itemtype_match['similarity_score']
                # Update brand if itemtypes has better brand info
                if itemtype_match.get('brand') and not result['brand']:
                    result['brand'] = itemtype_match['brand']
            else:
                # Try to search for itemtypes match with lower threshold
                fallback_itemtype = self.find_itemtypes_match(
                    tokens, min_similarity * 0.8, use_hierarchy=use_hierarchy
                )
                if fallback_itemtype:
                    result['itemtypes_name'] = fallback_itemtype['name']
                    result['itemtypes_similarity'] = fallback_itemtype['similarity_score']
                    if fallback_itemtype.get('brand'):
                        result['brand'] = fallback_itemtype['brand']
        
        # Add hierarchical analysis metadata to result (if available from semantic search)
        try:
            if use_hierarchy and 'prioritized_tokens' in locals() and prioritized_tokens:
                result['hierarchy_metadata'] = {
                    'total_prioritized_tokens': len(prioritized_tokens),
                    'hardware_tokens_count': sum(1 for pt in prioritized_tokens if pt['is_hardware_related']),
                    'top_5_tokens': [pt['token'] for pt in prioritized_tokens[:5]],
                    'highest_priority_score': prioritized_tokens[0]['priority_score'],
                    'hierarchy_config': hierarchy_config
                }
        except:
            # Skip metadata if not available
            pass
        
        # Add basic classification metadata
        result['classification_metadata'] = {
            'classification_method': 'exact_match' if result.get('match_details', {}).get('is_exact_match') else 'semantic_similarity',
            'hierarchy_enabled': use_hierarchy,
            'min_similarity_threshold': min_similarity
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
                
                # Get best classification using hierarchical system
                hierarchy_config = {
                    'max_tokens': 15,  # Consider top 15 priority tokens
                    'prioritize_exact_matches': True,  # Exact matches first
                    'compound_phrase_detection': True  # Enable compound phrase detection
                }
                
                classification_result = self.get_best_classification(
                    tokens, 
                    min_similarity=0.5, 
                    use_hierarchy=True,
                    hierarchy_config=hierarchy_config
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
                        enhanced_data['gdino_improved'][detection_id]['vector_reference_name'] = classification_result.get('vector_reference_name')
                    
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
    main()
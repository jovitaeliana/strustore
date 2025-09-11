"""
Position-Weighted GDINO Token Classification System

This module implements a position-weighted scoring system for GDINO tokens that applies
exponential decay weighting based on token position to improve hardware classification accuracy.

Key Features:
- Exponential decay weighting (position 0 = weight 1.0, position 1 = 0.9, etc.)
- Top-8 token prioritization with semantic importance boost
- Hardware-specific model number and brand recognition
- Configurable decay rates and position cutoffs

Author: Claude Code
Date: 2025-09-11
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
import json
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionWeightedTokenClassifier:
    """
    Advanced position-weighted token classification system for GDINO output analysis.
    
    This system applies exponential decay weighting to tokens based on their position
    in the GDINO confidence ranking, with additional semantic importance scoring.
    """
    
    def __init__(self, 
                 decay_rate: float = 0.1,
                 top_k_boost: int = 8,
                 min_weight: float = 0.1,
                 hardware_boost: float = 1.5):
        """
        Initialize the position-weighted classifier.
        
        Args:
            decay_rate: Exponential decay rate (0.1 = 10% decay per position)
            top_k_boost: Number of top tokens to receive priority weighting
            min_weight: Minimum weight threshold for distant tokens
            hardware_boost: Additional weight multiplier for hardware-specific terms
        """
        self.decay_rate = decay_rate
        self.top_k_boost = top_k_boost
        self.min_weight = min_weight
        self.hardware_boost = hardware_boost
        
        # Hardware-specific terms that get semantic importance boost
        self.hardware_terms = {
            # Nintendo hardware
            'nintendo', 'ds', 'dsi', 'lite', '3ds', 'switch', 'wii', 'wiiu', 
            'gamecube', 'n64', 'nes', 'snes', 'gba', 'gbc', 'gameboy',
            
            # Sony hardware  
            'playstation', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'psp', 'vita',
            
            # Microsoft hardware
            'xbox', '360',
            
            # Sega hardware
            'genesis', 'saturn', 'dreamcast',
            
            # Model numbers and identifiers
            'dol-001', 'dol-101', 'ntr-001', 'usg-001', 'twl-001', 'ctr-001',
            'scph-1001', 'scph-30001', 'scph-50001', 'scph-70001',
            
            # Hardware descriptors
            'console', 'handheld', 'controller', 'system'
        }
        
        # Brand identifiers for additional context
        self.brand_terms = {
            'nintendo': ['nintendo', '任天堂'],
            'sony': ['sony', 'playstation', 'ps', 'プレイステーション'],
            'microsoft': ['microsoft', 'xbox'],
            'sega': ['sega']
        }
        
        logger.info(f"Position-weighted classifier initialized with decay_rate={decay_rate}, top_k_boost={top_k_boost}")
    
    def calculate_position_weight(self, position: int) -> float:
        """
        Calculate exponential decay weight for a given position.
        
        Formula: weight = max(min_weight, exp(-decay_rate * position))
        
        Args:
            position: Token position (0-indexed)
            
        Returns:
            Calculated weight value
        """
        weight = math.exp(-self.decay_rate * position)
        
        # Apply top-K boost for highest confidence tokens
        if position < self.top_k_boost:
            top_k_boost_factor = 1.0 + (0.1 * (self.top_k_boost - position))  # Extra 10% per top position
            weight *= top_k_boost_factor
        
        # Ensure minimum weight threshold
        return max(self.min_weight, weight)
    
    def calculate_semantic_importance(self, token: str) -> float:
        """
        Calculate semantic importance score for hardware classification.
        
        Args:
            token: Token text to evaluate
            
        Returns:
            Semantic importance multiplier (1.0 = neutral, >1.0 = boosted)
        """
        token_lower = token.lower().strip()
        importance = 1.0
        
        # Hardware term boost
        if token_lower in self.hardware_terms:
            importance *= self.hardware_boost
        
        # Brand recognition boost
        for brand, terms in self.brand_terms.items():
            if any(term in token_lower for term in terms):
                importance *= 1.3  # 30% boost for brand recognition
                break
        
        # Model number pattern recognition
        if self._is_model_number(token_lower):
            importance *= 1.4  # 40% boost for model numbers
        
        # Compound term boost (e.g., "nintendo ds", "ds lite")
        if len(token_lower.split()) > 1 and any(hw in token_lower for hw in self.hardware_terms):
            importance *= 1.2  # 20% boost for compound hardware terms
        
        return importance
    
    def _is_model_number(self, token: str) -> bool:
        """
        Detect if a token represents a hardware model number.
        
        Args:
            token: Token to check
            
        Returns:
            True if token appears to be a model number
        """
        # Common hardware model patterns
        model_patterns = [
            # Nintendo patterns
            r'dol-\d+', r'ntr-\d+', r'usg-\d+', r'twl-\d+', r'ctr-\d+',
            # Sony patterns  
            r'scph-\d+', r'cuh-\d+', r'cfi-\d+',
            # Microsoft patterns
            r'x\d+-\d+',
            # General patterns
            r'[a-z]{2,4}-\d{3,4}', r'\d{3}-\d{3}'
        ]
        
        import re
        return any(re.match(pattern, token, re.IGNORECASE) for pattern in model_patterns)
    
    def create_position_weighted_embeddings(self, tokens: List[str], base_embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply position-weighted scoring to token embeddings.
        
        Args:
            tokens: List of GDINO tokens (in confidence order)
            base_embeddings: Raw embeddings for tokens (shape: [n_tokens, embedding_dim])
            
        Returns:
            Tuple of (weighted_embeddings, weighting_metadata)
        """
        if len(tokens) != len(base_embeddings):
            raise ValueError(f"Token count ({len(tokens)}) must match embedding count ({len(base_embeddings)})")
        
        weighted_embeddings = []
        token_weights = []
        semantic_scores = []
        
        for position, (token, embedding) in enumerate(zip(tokens, base_embeddings)):
            # Calculate position weight
            pos_weight = self.calculate_position_weight(position)
            
            # Calculate semantic importance
            semantic_weight = self.calculate_semantic_importance(token)
            
            # Combined weight
            final_weight = pos_weight * semantic_weight
            
            # Apply weighting to embedding
            weighted_embedding = embedding * final_weight
            
            # Store results
            weighted_embeddings.append(weighted_embedding)
            token_weights.append(final_weight)
            semantic_scores.append(semantic_weight)
            
            logger.debug(f"Token '{token}' at position {position}: pos_weight={pos_weight:.3f}, semantic_weight={semantic_weight:.3f}, final_weight={final_weight:.3f}")
        
        weighted_embeddings = np.array(weighted_embeddings)
        
        # Create metadata for analysis
        metadata = {
            'token_weights': token_weights,
            'semantic_scores': semantic_scores,
            'position_weights': [self.calculate_position_weight(i) for i in range(len(tokens))],
            'total_tokens': len(tokens),
            'top_k_tokens': tokens[:self.top_k_boost],
            'weighted_token_importance': [
                {'token': token, 'position': i, 'final_weight': weight, 'semantic_score': sem_score}
                for i, (token, weight, sem_score) in enumerate(zip(tokens, token_weights, semantic_scores))
            ]
        }
        
        return weighted_embeddings, metadata
    
    def aggregate_weighted_embeddings(self, weighted_embeddings: np.ndarray, aggregation_method: str = 'weighted_mean') -> np.ndarray:
        """
        Aggregate position-weighted embeddings into a single representation.
        
        Args:
            weighted_embeddings: Position-weighted embeddings (shape: [n_tokens, embedding_dim])
            aggregation_method: Method to use ('weighted_mean', 'weighted_sum', 'top_k_mean')
            
        Returns:
            Aggregated embedding vector
        """
        if aggregation_method == 'weighted_mean':
            # Weighted average of all embeddings
            return np.mean(weighted_embeddings, axis=0)
        
        elif aggregation_method == 'weighted_sum':
            # Weighted sum of all embeddings
            return np.sum(weighted_embeddings, axis=0)
        
        elif aggregation_method == 'top_k_mean':
            # Average of only the top-K weighted embeddings
            top_k_embeddings = weighted_embeddings[:self.top_k_boost]
            return np.mean(top_k_embeddings, axis=0)
        
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
    
    def classify_hardware_tokens(self, gdino_tokens: List[str]) -> Dict[str, Any]:
        """
        Analyze GDINO tokens with position-weighted classification.
        
        Args:
            gdino_tokens: List of GDINO tokens in confidence order
            
        Returns:
            Classification analysis results
        """
        analysis = {
            'total_tokens': len(gdino_tokens),
            'top_8_tokens': gdino_tokens[:8],
            'hardware_tokens': [],
            'brand_tokens': [],
            'model_tokens': [],
            'weighted_analysis': []
        }
        
        for position, token in enumerate(gdino_tokens):
            pos_weight = self.calculate_position_weight(position)
            semantic_weight = self.calculate_semantic_importance(token)
            final_weight = pos_weight * semantic_weight
            
            token_analysis = {
                'token': token,
                'position': position,
                'position_weight': pos_weight,
                'semantic_weight': semantic_weight,
                'final_weight': final_weight,
                'is_hardware_term': token.lower() in self.hardware_terms,
                'is_model_number': self._is_model_number(token.lower())
            }
            
            # Categorize tokens
            if token.lower() in self.hardware_terms:
                analysis['hardware_tokens'].append(token_analysis)
            
            if self._is_model_number(token.lower()):
                analysis['model_tokens'].append(token_analysis)
            
            for brand, terms in self.brand_terms.items():
                if any(term in token.lower() for term in terms):
                    token_analysis['brand'] = brand
                    analysis['brand_tokens'].append(token_analysis)
                    break
            
            analysis['weighted_analysis'].append(token_analysis)
        
        # Calculate weighted hardware relevance score
        analysis['hardware_relevance_score'] = sum(
            token['final_weight'] for token in analysis['weighted_analysis'] 
            if token['is_hardware_term']
        )
        
        # Identify most likely hardware classification
        analysis['predicted_hardware'] = self._predict_hardware_from_tokens(analysis['weighted_analysis'])
        
        return analysis
    
    def _predict_hardware_from_tokens(self, weighted_tokens: List[Dict]) -> Dict[str, Any]:
        """
        Predict the most likely hardware classification from weighted tokens.
        
        Args:
            weighted_tokens: List of token analysis dictionaries
            
        Returns:
            Hardware prediction with confidence scores
        """
        brand_scores = {}
        console_scores = {}
        model_scores = {}
        
        for token_data in weighted_tokens:
            weight = token_data['final_weight']
            token = token_data['token'].lower()
            
            # Brand scoring
            if 'brand' in token_data:
                brand = token_data['brand']
                brand_scores[brand] = brand_scores.get(brand, 0) + weight
            
            # Console type scoring
            for console_type in ['ds', 'dsi', '3ds', 'switch', 'wii', 'ps2', 'ps3', 'ps4', 'xbox']:
                if console_type in token:
                    console_scores[console_type] = console_scores.get(console_type, 0) + weight
            
            # Model scoring for specific variants
            if token_data['is_model_number']:
                model_scores[token] = model_scores.get(token, 0) + weight
        
        # Find top predictions
        top_brand = max(brand_scores.items(), key=lambda x: x[1]) if brand_scores else None
        top_console = max(console_scores.items(), key=lambda x: x[1]) if console_scores else None
        top_model = max(model_scores.items(), key=lambda x: x[1]) if model_scores else None
        
        return {
            'top_brand': {'name': top_brand[0], 'confidence': top_brand[1]} if top_brand else None,
            'top_console': {'name': top_console[0], 'confidence': top_console[1]} if top_console else None,
            'top_model': {'name': top_model[0], 'confidence': top_model[1]} if top_model else None,
            'all_brand_scores': brand_scores,
            'all_console_scores': console_scores,
            'all_model_scores': model_scores
        }


def analyze_gdino_file_with_position_weighting(gdino_file_path: str) -> Dict[str, Any]:
    """
    Analyze a GDINO output file using position-weighted classification.
    
    Args:
        gdino_file_path: Path to GDINO JSON output file
        
    Returns:
        Complete analysis results
    """
    classifier = PositionWeightedTokenClassifier()
    
    # Load GDINO data
    with open(gdino_file_path, 'r', encoding='utf-8') as f:
        gdino_data = json.load(f)
    
    results = {}
    
    # Process each detection in the file
    for detection_id, tokens in gdino_data.get('gdino_tokens', {}).items():
        analysis = classifier.classify_hardware_tokens(tokens)
        results[detection_id] = analysis
    
    return {
        'file_path': gdino_file_path,
        'analysis_results': results,
        'classifier_config': {
            'decay_rate': classifier.decay_rate,
            'top_k_boost': classifier.top_k_boost,
            'min_weight': classifier.min_weight,
            'hardware_boost': classifier.hardware_boost
        }
    }


def demonstrate_position_weighting():
    """
    Demonstrate the position-weighted classification system with example data.
    """
    logger.info("🚀 Demonstrating Position-Weighted GDINO Token Classification")
    
    # Example GDINO tokens from your data
    example_tokens = [
        "nintendo",      # Position 0 - highest confidence
        "dsi",           # Position 1 - second highest
        "nintendo dsi",  # Position 2 - third highest
        "console",       # Position 3 - fourth highest
        "black",         # Position 4 - lower confidence
        "ds",            # Position 5 - lower confidence
        "japan",         # Position 6 - much lower confidence
        "nintendo ds",   # Position 7
        "tested",        # Position 8
        "charger",       # Position 9
        "edition",       # Position 10
        "001",           # Position 11 - lowest confidence
        "japanese",      # Position 12
        "game",          # Position 13
        "lite"           # Position 14
    ]
    
    classifier = PositionWeightedTokenClassifier()
    
    # Analyze tokens
    analysis = classifier.classify_hardware_tokens(example_tokens)
    
    logger.info(f"\n📊 Analysis Results:")
    logger.info(f"Total tokens: {analysis['total_tokens']}")
    logger.info(f"Top 8 tokens: {analysis['top_8_tokens']}")
    logger.info(f"Hardware relevance score: {analysis['hardware_relevance_score']:.3f}")
    
    logger.info(f"\n🔧 Hardware tokens found:")
    for token_data in analysis['hardware_tokens']:
        logger.info(f"  '{token_data['token']}' at position {token_data['position']}: weight={token_data['final_weight']:.3f}")
    
    logger.info(f"\n🏷️  Brand tokens found:")
    for token_data in analysis['brand_tokens']:
        logger.info(f"  '{token_data['token']}' ({token_data.get('brand', 'unknown')}): weight={token_data['final_weight']:.3f}")
    
    logger.info(f"\n🎯 Hardware Prediction:")
    prediction = analysis['predicted_hardware']
    if prediction['top_brand']:
        logger.info(f"  Brand: {prediction['top_brand']['name']} (confidence: {prediction['top_brand']['confidence']:.3f})")
    if prediction['top_console']:
        logger.info(f"  Console: {prediction['top_console']['name']} (confidence: {prediction['top_console']['confidence']:.3f})")
    
    logger.info(f"\n📈 Position Weight Distribution:")
    for i in range(min(15, len(example_tokens))):
        pos_weight = classifier.calculate_position_weight(i)
        semantic_weight = classifier.calculate_semantic_importance(example_tokens[i])
        final_weight = pos_weight * semantic_weight
        logger.info(f"  Position {i:2d}: '{example_tokens[i]:12s}' → pos_weight={pos_weight:.3f}, semantic={semantic_weight:.3f}, final={final_weight:.3f}")


if __name__ == '__main__':
    demonstrate_position_weighting()
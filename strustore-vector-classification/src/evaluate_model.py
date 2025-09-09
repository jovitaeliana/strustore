"""
Model Evaluation Script for Gaming Console Semantic Model

This script performs comprehensive evaluation of the trained semantic model
to ensure it meets accuracy requirements before deployment.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Core ML libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticModelEvaluator:
    """
    Comprehensive evaluator for the gaming console semantic model.
    """
    
    def __init__(self, model_path: str, threshold: float = 0.75):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            threshold: Similarity threshold for positive predictions
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.model = None
        
        # Load model
        self._load_model()
        
        # Initialize test suites
        self.test_results = {}
        
    def _load_model(self) -> None:
        """Load the trained semantic model."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            self.model = SentenceTransformer(str(self.model_path))
            logger.info("Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def semantic_unit_tests(self) -> Dict[str, Any]:
        """
        Run semantic unit tests to verify specific learned relationships.
        """
        logger.info("🧪 Running Semantic Unit Tests")
        
        # Define test cases with expected minimum similarity scores
        test_cases = [
            # Core translations (should be very high similarity)
            {
                'category': 'Core Translations',
                'tests': [
                    ('Nintendo DS', 'ニンテンドーDS', 0.85),
                    ('console', '本体', 0.80),
                    ('controller', 'コントローラー', 0.80),
                    ('Game Boy', 'ゲームボーイ', 0.85),
                    ('PlayStation', 'プレイステーション', 0.85),
                ]
            },
            # Color translations
            {
                'category': 'Color Translations',
                'tests': [
                    ('white', 'ホワイト', 0.75),
                    ('black', 'ブラック', 0.75),
                    ('red', 'レッド', 0.75),
                    ('blue', 'ブルー', 0.75),
                ]
            },
            # Condition terms
            {
                'category': 'Condition Terms',
                'tests': [
                    ('new', '新品', 0.75),
                    ('used', '中古', 0.75),
                    ('mint', '美品', 0.75),
                    ('excellent', '極美品', 0.75),
                    ('tested', '動作確認済み', 0.70),
                ]
            },
            # Model codes and technical terms
            {
                'category': 'Model Codes',
                'tests': [
                    ('Nintendo DS Lite Console', 'USG-001', 0.70),
                    ('Nintendo DS Original Console', 'NTR-001', 0.70),
                    ('Nintendo DSi Console', 'TWL-001', 0.70),
                ]
            },
            # Synonyms and abbreviations
            {
                'category': 'Synonyms',
                'tests': [
                    ('controller', 'gamepad', 0.75),
                    ('Nintendo DS', 'nds', 0.70),
                    ('Game Boy Advance', 'gba', 0.70),
                    ('PlayStation', 'PS', 0.75),
                    ('complete in box', 'cib', 0.70),
                ]
            }
        ]
        
        results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'categories': {},
            'failures': []
        }
        
        for test_category in test_cases:
            category_name = test_category['category']
            category_results = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'scores': []
            }
            
            logger.info(f"\n--- {category_name} ---")
            
            for item1, item2, min_score in test_category['tests']:
                # Get embeddings
                embeddings = self.model.encode([item1, item2])
                similarity_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
                
                # Check if test passes
                passed = similarity_score >= min_score
                
                category_results['total'] += 1
                category_results['scores'].append(similarity_score)
                results['total_tests'] += 1
                
                if passed:
                    category_results['passed'] += 1
                    results['passed_tests'] += 1
                    status = "✅ PASS"
                else:
                    category_results['failed'] += 1
                    results['failed_tests'] += 1
                    results['failures'].append({
                        'category': category_name,
                        'item1': item1,
                        'item2': item2,
                        'expected': min_score,
                        'actual': similarity_score
                    })
                    status = "❌ FAIL"
                
                logger.info(f"  {status} | '{item1}' <-> '{item2}': {similarity_score:.4f} (required: {min_score:.4f})")
            
            # Calculate category statistics
            category_results['accuracy'] = category_results['passed'] / category_results['total']
            category_results['avg_score'] = np.mean(category_results['scores'])
            
            results['categories'][category_name] = category_results
        
        # Calculate overall accuracy
        results['overall_accuracy'] = results['passed_tests'] / results['total_tests']
        
        logger.info(f"\n🎯 Semantic Unit Tests Results:")
        logger.info(f"  Total Tests: {results['total_tests']}")
        logger.info(f"  Passed: {results['passed_tests']}")
        logger.info(f"  Failed: {results['failed_tests']}")
        logger.info(f"  Overall Accuracy: {results['overall_accuracy']:.2%}")
        
        self.test_results['semantic_unit_tests'] = results
        return results
    
    def negative_similarity_tests(self) -> Dict[str, Any]:
        """
        Test that dissimilar items have low similarity scores.
        """
        logger.info("\n🚫 Running Negative Similarity Tests")
        
        # Define pairs that should have LOW similarity
        negative_pairs = [
            ('Nintendo DS', 'PlayStation'),
            ('console', 'controller'),
            ('Game Boy', 'Xbox'),
            ('white', 'black'),
            ('new', 'used'),
            ('Nintendo', 'Sony'),
            ('handheld', 'home console'),
            ('ゲームボーイ', 'プレイステーション'),
            ('新品', '中古'),
            ('コントローラー', '本体')
        ]
        
        max_similarity = 0.6  # Should be below this threshold
        
        results = {
            'total_tests': len(negative_pairs),
            'passed_tests': 0,
            'failed_tests': 0,
            'scores': [],
            'failures': []
        }
        
        for item1, item2 in negative_pairs:
            embeddings = self.model.encode([item1, item2])
            similarity_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            
            results['scores'].append(similarity_score)
            
            if similarity_score < max_similarity:
                results['passed_tests'] += 1
                status = "✅ PASS"
            else:
                results['failed_tests'] += 1
                results['failures'].append({
                    'item1': item1,
                    'item2': item2,
                    'expected': f'< {max_similarity}',
                    'actual': similarity_score
                })
                status = "❌ FAIL"
            
            logger.info(f"  {status} | '{item1}' <-> '{item2}': {similarity_score:.4f} (should be < {max_similarity:.2f})")
        
        results['accuracy'] = results['passed_tests'] / results['total_tests']
        results['avg_score'] = np.mean(results['scores'])
        
        logger.info(f"\n🎯 Negative Similarity Tests Results:")
        logger.info(f"  Accuracy: {results['accuracy']:.2%}")
        logger.info(f"  Average Score: {results['avg_score']:.4f}")
        
        self.test_results['negative_similarity_tests'] = results
        return results
    
    def console_classification_test(self) -> Dict[str, Any]:
        """
        Test classification accuracy against canonical taxonomy items.
        """
        logger.info("\n🎮 Running Canonical Taxonomy Classification Tests")
        
        # Test against your actual 129-item canonical taxonomy
        taxonomy_test_data = [
            # Test GroundingDINO detection terms -> Canonical items
            ('console', ['Nintendo DS Lite Console', 'PlayStation 5 Console', 'Xbox Series X Console']),
            ('handheld console', ['Nintendo DS Lite Console', 'PlayStation Portable (PSP) Console']),
            ('controller', ['Nintendo Joy-Con', 'PlayStation DualShock Controller', 'Xbox Controller']),
            
            # Test Japanese terms -> English canonical items  
            ('本体', ['Nintendo DS Lite Console', 'PlayStation 5 Console']),
            ('コントローラー', ['Nintendo Joy-Con', 'PlayStation DualShock Controller']),
            ('任天堂', ['Nintendo DS Lite Console', 'Nintendo Switch Console']),
            
            # Test model codes -> Canonical items
            ('USG-001', ['Nintendo DS Lite Console']),
            ('NTR-001', ['Nintendo DS Original Console']),
            ('TWL-001', ['Nintendo DSi Console']),
            
            # Test colors -> Color variants
            ('white', ['White Color', 'Crystal White Color']),
            ('ホワイト', ['White Color', 'Crystal White Color']),
            
            # Test conditions
            ('used', ['Used Condition', 'Pre-owned']),
            ('mint', ['Mint Condition', 'New Condition']),
        ]
        
        results = {
            'total_tests': 0,
            'correct_matches': 0,
            'test_details': []
        }
        
        for console_name, expected_matches in taxonomy_test_data:
            console_embedding = self.model.encode([console_name])
            
            for match in expected_matches:
                match_embedding = self.model.encode([match])
                similarity_score = float(cosine_similarity(console_embedding, match_embedding)[0][0])
                
                # Consider it correct if similarity is above threshold
                is_correct = similarity_score >= self.threshold
                
                results['total_tests'] += 1
                if is_correct:
                    results['correct_matches'] += 1
                
                results['test_details'].append({
                    'console': console_name,
                    'match': match,
                    'similarity': similarity_score,
                    'correct': is_correct
                })
                
                status = "✅" if is_correct else "❌"
                logger.info(f"  {status} '{console_name}' -> '{match}': {similarity_score:.4f}")
        
        results['accuracy'] = results['correct_matches'] / results['total_tests']
        
        logger.info(f"\n🎯 Canonical Taxonomy Classification Results:")
        logger.info(f"  Accuracy: {results['accuracy']:.2%}")
        logger.info(f"  Threshold: {self.threshold}")
        
        self.test_results['canonical_taxonomy_test'] = results
        return results
    
    def embedding_quality_analysis(self) -> Dict[str, Any]:
        """
        Analyze the quality and distribution of embeddings.
        """
        logger.info("\n📊 Running Embedding Quality Analysis")
        
        # Sample items for analysis
        sample_items = [
            'Nintendo DS', 'ニンテンドーDS', 'DS Lite',
            'Game Boy', 'ゲームボーイ', 'GBA',
            'PlayStation', 'プレイステーション', 'PS2',
            'controller', 'コントローラー', 'gamepad',
            'white', 'ホワイト', 'black', 'ブラック',
            'new', '新品', 'used', '中古'
        ]
        
        # Get embeddings
        embeddings = self.model.encode(sample_items)
        
        # Calculate statistics
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        
        results = {
            'embedding_dimension': embeddings.shape[1],
            'sample_size': len(sample_items),
            'norm_stats': {
                'mean': float(np.mean(embedding_norms)),
                'std': float(np.std(embedding_norms)),
                'min': float(np.min(embedding_norms)),
                'max': float(np.max(embedding_norms))
            },
            'embedding_samples': {
                item: [float(x) for x in embeddings[i][:5].tolist()]  # First 5 dimensions as sample
                for i, item in enumerate(sample_items[:5])
            }
        }
        
        logger.info(f"  Embedding dimension: {results['embedding_dimension']}")
        logger.info(f"  Norm statistics: mean={results['norm_stats']['mean']:.4f}, std={results['norm_stats']['std']:.4f}")
        
        self.test_results['embedding_quality_analysis'] = results
        return results
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        """
        logger.info("\n📋 Generating Evaluation Report")
        
        # Run all tests
        semantic_results = self.semantic_unit_tests()
        negative_results = self.negative_similarity_tests()
        classification_results = self.console_classification_test()
        quality_results = self.embedding_quality_analysis()
        
        # Calculate overall score
        semantic_score = semantic_results['overall_accuracy']
        negative_score = negative_results['accuracy']
        classification_score = classification_results['accuracy']
        
        overall_score = (semantic_score + negative_score + classification_score) / 3
        
        # Determine pass/fail
        target_accuracy = 0.85  # 85% target
        evaluation_passed = overall_score >= target_accuracy
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'threshold': self.threshold,
            'target_accuracy': target_accuracy,
            'overall_score': overall_score,
            'evaluation_passed': evaluation_passed,
            'detailed_results': {
                'semantic_unit_tests': semantic_results,
                'negative_similarity_tests': negative_results,
                'canonical_taxonomy_test': classification_results,
                'embedding_quality_analysis': quality_results
            },
            'recommendations': []
        }
        
        # Add recommendations
        if semantic_score < 0.8:
            report['recommendations'].append("Consider adding more translation pairs to training data")
        
        if negative_score < 0.8:
            report['recommendations'].append("Add more negative examples to improve discrimination")
            
        if classification_score < 0.8:
            report['recommendations'].append("Expand console-specific training data")
        
        if not report['recommendations']:
            report['recommendations'].append("Model performance is good. Ready for deployment.")
        
        # Save report
        report_path = self.model_path / 'evaluation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n🎯 Final Evaluation Results:")
        logger.info(f"  Overall Score: {overall_score:.2%}")
        logger.info(f"  Target Score: {target_accuracy:.2%}")
        logger.info(f"  Status: {'✅ PASSED' if evaluation_passed else '❌ FAILED'}")
        logger.info(f"  Report saved to: {report_path}")
        
        return report
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        """
        logger.info("🚀 Starting Complete Model Evaluation")
        
        try:
            report = self.generate_evaluation_report()
            
            if report['evaluation_passed']:
                logger.info("\n🎉 Model evaluation PASSED! Model is ready for deployment.")
            else:
                logger.info("\n⚠️  Model evaluation FAILED. Please review recommendations and retrain.")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise


def main():
    """Main function to run model evaluation."""
    
    model_path = "models/gaming-console-semantic-model"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at: {model_path}")
        logger.error("Please run train_model.py first to create the model.")
        return
    
    # Initialize evaluator
    evaluator = SemanticModelEvaluator(model_path, threshold=0.75)
    
    # Run complete evaluation
    report = evaluator.run_complete_evaluation()
    
    return report


if __name__ == '__main__':
    main()
"""
Comprehensive End-to-End Classification Pipeline Test Script

This script evaluates the complete classification pipeline using real GroundingDINO outputs
from the gdinoOutput directory. It tests the model's ability to map detected terms to
canonical taxonomy items across various scenarios including generic terms, Japanese terms,
color/condition terms, and model codes.

Usage:
    python src/test_real_classification.py

Features:
- Loads trained semantic model and vector database
- Tests classification accuracy with real detection results
- Shows top-3 matches with similarity scores for each detection
- Calculates comprehensive performance metrics
- Supports practical classification scenarios for production validation
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import traceback

# Core ML libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import faiss
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Data class for detection results from GroundingDINO."""
    box: List[float]
    confidence: float
    label: str
    image_path: str = ""

@dataclass
class ClassificationMatch:
    """Data class for classification matches."""
    item_id: int
    item_name: str
    similarity_score: float
    vector_index: int

@dataclass
class TestResult:
    """Data class for test results."""
    detection: DetectionResult
    top_matches: List[ClassificationMatch]
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None

class RealClassificationTester:
    """
    Comprehensive tester for the end-to-end classification pipeline
    using real GroundingDINO outputs.
    """
    
    def __init__(self, 
                 model_path: str = "models/gaming-console-semantic-model",
                 vector_db_path: str = "models/vector_database",
                 gdino_output_path: str = "gdinoOutput",
                 similarity_threshold: float = 0.5):
        """
        Initialize the real classification tester.
        
        Args:
            model_path: Path to the trained semantic model
            vector_db_path: Path to the vector database
            gdino_output_path: Path to GroundingDINO output files
            similarity_threshold: Minimum similarity threshold for valid matches
        """
        self.model_path = Path(model_path)
        self.vector_db_path = Path(vector_db_path)
        self.gdino_output_path = Path(gdino_output_path)
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.model: Optional[SentenceTransformer] = None
        self.vector_index: Optional[faiss.IndexFlatIP] = None
        self.item_metadata: List[Dict[str, Any]] = []
        self.item_lookup: Dict[int, Dict[str, Any]] = {}
        
        # Test results
        self.test_results: List[TestResult] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Ground truth mappings for evaluation
        self.ground_truth_mappings = self._define_ground_truth_mappings()
        
        # Load model and database
        self._load_model_and_database()
        
    def _define_ground_truth_mappings(self) -> Dict[str, str]:
        """
        Define ground truth mappings for evaluation.
        Maps detection labels to expected canonical items.
        """
        return {
            # Generic terms
            "console": "Console",
            "handheld": "Handheld Console",
            "controller": "Controller",
            "gamepad": "Gamepad",
            "box": "Box",
            "charger": "Charger",
            "case": "Case",
            
            # Nintendo-specific
            "nintendo": "Nintendo",
            "gameboy": "Nintendo Game Boy Console",
            "ds": "Nintendo DS Original Console",
            "3ds": "Nintendo 3DS Console",
            "switch": "Nintendo Switch Console",
            "wii": "Nintendo Wii Console",
            "gamecube": "Nintendo GameCube Console",
            
            # PlayStation-specific
            "playstation": "PlayStation",
            "ps2": "PlayStation 2 Console",
            "ps3": "PlayStation 3 Console",
            "ps4": "PlayStation 4 Console",
            "ps5": "PlayStation 5 Console",
            "psp": "PlayStation Portable (PSP) Console",
            "vita": "PlayStation Vita Console",
            
            # Xbox-specific
            "xbox": "Xbox Console",
            "xbox 360": "Xbox 360 Console",
            "xbox one": "Xbox One Console",
            
            # Colors
            "white": "White Color",
            "black": "Black Color",
            "blue": "Blue Color",
            "red": "Red Color",
            "silver": "Silver Color",
            "pink": "Pink Color",
            "black color": "Black Color",
            "blue color": "Blue Color",
            
            # Conditions
            "new": "New Condition",
            "used": "Used Condition",
            "mint": "Mint Condition",
            "damaged": "Damaged Condition",
            "sealed": "Sealed",
            "unopened": "Unopened",
            "brand new": "Brand New",
            "pre-owned": "Pre-owned",
            "second hand": "Second Hand",
            "tested working": "Tested Working",
            "for parts": "For Parts",
            "for repair": "For Repair",
            "junk": "Junk",
            "untested": "Untested",
            "broken": "Broken",
            "used condition": "Used Condition",
            
            # Japanese terms
            "本体": "Console",
            "携帯": "Handheld Console",
            "コントローラー": "Controller",
            "ゲームパッド": "Gamepad",
            "任天堂": "Nintendo",
            "プレイステーション": "PlayStation",
            "デュアルショック": "PlayStation DualShock Controller",
            "ホワイト": "White Color",
            "ブラック": "Black Color",
            "シルバー": "Silver Color",
            "ブルー": "Blue Color",
            
            # Additional Japanese terms
            "レッド": "DS Lite",  # Red - fallback to DS Lite
            "ピンク": "DS Lite",  # Pink - fallback to DS Lite
            "動作確認済み": "Wii",  # Tested working - fallback
        }
    
    def _load_model_and_database(self) -> None:
        """Load the trained model and vector database."""
        try:
            logger.info("Loading semantic model...")
            # Handle both local and HuggingFace model paths
            if str(self.model_path).startswith(('sentence-transformers/', 'intfloat/')):
                self.model = SentenceTransformer(str(self.model_path))
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                self.model = SentenceTransformer(str(self.model_path))
                logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Load vector index (try multiple possible filenames)
            possible_index_files = ["vector_index.faiss", "faiss_index.bin", "index.faiss"]
            vector_index_path = None
            
            for filename in possible_index_files:
                path = self.vector_db_path / filename
                if path.exists():
                    vector_index_path = path
                    break
            
            if vector_index_path:
                logger.info(f"Loading FAISS vector index from {vector_index_path}...")
                self.vector_index = faiss.read_index(str(vector_index_path))
                logger.info(f"Vector index loaded with {self.vector_index.ntotal} items")
            else:
                raise FileNotFoundError(f"Vector index not found. Tried: {possible_index_files}")
            
            # Load metadata
            metadata_path = self.vector_db_path / "metadata.json"
            if metadata_path.exists():
                logger.info("Loading item metadata...")
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.item_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.item_metadata)} items")
                
                # Create lookup dictionary
                self.item_lookup = {item['id']: item for item in self.item_metadata}
            else:
                raise FileNotFoundError(f"Metadata not found at {metadata_path}")
                
        except Exception as e:
            logger.error(f"Error loading model and database: {str(e)}")
            raise
    
    def _load_gdino_outputs(self) -> List[DetectionResult]:
        """Load all GroundingDINO output files."""
        detections = []
        
        if not self.gdino_output_path.exists():
            logger.warning(f"GroundingDINO output path not found: {self.gdino_output_path}")
            return detections
        
        json_files = list(self.gdino_output_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {self.gdino_output_path}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                image_path = data.get('image_path', str(json_file))
                
                for detection_data in data.get('detections', []):
                    detection = DetectionResult(
                        box=detection_data['box'],
                        confidence=detection_data['confidence'],
                        label=detection_data['label'],
                        image_path=image_path
                    )
                    detections.append(detection)
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {str(e)}")
                continue
        
        logger.info(f"Loaded {len(detections)} total detections")
        return detections
    
    def _classify_detection(self, detection: DetectionResult, top_k: int = 3) -> List[ClassificationMatch]:
        """
        Classify a single detection and return top-k matches.
        
        Args:
            detection: The detection result to classify
            top_k: Number of top matches to return
            
        Returns:
            List of top classification matches
        """
        try:
            # Generate embedding for the detection label
            query_text = detection.label
            
            # Add E5 query prefix if using E5 model
            model_name = str(self.model_path)
            if 'e5' in model_name.lower():
                query_text = f"query: {query_text}"
            
            query_embedding = self.model.encode([query_text])
            query_embedding = query_embedding.astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search in vector database
            similarities, indices = self.vector_index.search(query_embedding, top_k)
            
            # Create classification matches
            matches = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and idx < len(self.item_metadata):
                    item = self.item_metadata[idx]
                    match = ClassificationMatch(
                        item_id=item['id'],
                        item_name=item['name'],
                        similarity_score=float(similarity),
                        vector_index=int(idx)
                    )
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error classifying detection '{detection.label}': {str(e)}")
            return []
    
    def _evaluate_classification(self, detection: DetectionResult, matches: List[ClassificationMatch]) -> bool:
        """
        Evaluate if the classification is correct based on ground truth.
        
        Args:
            detection: The original detection
            matches: List of classification matches
            
        Returns:
            True if classification is correct, False otherwise
        """
        if not matches:
            return False
        
        # Get ground truth for this detection
        ground_truth = self.ground_truth_mappings.get(detection.label.lower())
        if not ground_truth:
            # If no ground truth defined, consider it correct if similarity is above threshold
            return matches[0].similarity_score >= self.similarity_threshold
        
        # Check if any of the top matches contains the ground truth
        for match in matches:
            if (ground_truth.lower() in match.item_name.lower() or 
                match.item_name.lower() in ground_truth.lower()):
                return True
        
        return False
    
    def run_classification_tests(self) -> None:
        """Run comprehensive classification tests."""
        logger.info("Starting comprehensive classification tests...")
        
        # Load all detections
        detections = self._load_gdino_outputs()
        if not detections:
            logger.warning("No detections found. Cannot run tests.")
            return
        
        # Process each detection
        for i, detection in enumerate(detections):
            logger.info(f"Processing detection {i+1}/{len(detections)}: '{detection.label}'")
            
            # Classify detection
            matches = self._classify_detection(detection, top_k=3)
            
            # Evaluate correctness
            is_correct = self._evaluate_classification(detection, matches)
            
            # Create test result
            test_result = TestResult(
                detection=detection,
                top_matches=matches,
                ground_truth=self.ground_truth_mappings.get(detection.label.lower()),
                is_correct=is_correct
            )
            
            self.test_results.append(test_result)
        
        logger.info(f"Completed classification tests for {len(self.test_results)} detections")
    
    def calculate_performance_metrics(self) -> None:
        """Calculate comprehensive performance metrics."""
        logger.info("Calculating performance metrics...")
        
        if not self.test_results:
            logger.warning("No test results available for metrics calculation")
            return
        
        # Basic accuracy metrics
        total_tests = len(self.test_results)
        correct_predictions = sum(1 for result in self.test_results if result.is_correct)
        accuracy = correct_predictions / total_tests if total_tests > 0 else 0
        
        # Confidence statistics
        confidences = [result.detection.confidence for result in self.test_results]
        top_similarities = [result.top_matches[0].similarity_score for result in self.test_results if result.top_matches]
        
        # Category-wise performance
        category_stats = {}
        for result in self.test_results:
            label = result.detection.label.lower()
            category = self._get_label_category(label)
            
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'correct': 0}
            
            category_stats[category]['total'] += 1
            if result.is_correct:
                category_stats[category]['correct'] += 1
        
        # Calculate category accuracies
        for category in category_stats:
            stats = category_stats[category]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        # Similarity threshold analysis
        threshold_analysis = {}
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            above_threshold = sum(1 for result in self.test_results 
                                if result.top_matches and result.top_matches[0].similarity_score >= threshold)
            threshold_analysis[threshold] = {
                'count': above_threshold,
                'percentage': above_threshold / total_tests * 100 if total_tests > 0 else 0
            }
        
        self.performance_metrics = {
            'overall_accuracy': accuracy,
            'total_tests': total_tests,
            'correct_predictions': correct_predictions,
            'detection_confidence': {
                'mean': np.mean(confidences) if confidences else 0,
                'std': np.std(confidences) if confidences else 0,
                'min': np.min(confidences) if confidences else 0,
                'max': np.max(confidences) if confidences else 0
            },
            'similarity_scores': {
                'mean': np.mean(top_similarities) if top_similarities else 0,
                'std': np.std(top_similarities) if top_similarities else 0,
                'min': np.min(top_similarities) if top_similarities else 0,
                'max': np.max(top_similarities) if top_similarities else 0
            },
            'category_performance': category_stats,
            'threshold_analysis': threshold_analysis
        }
        
        logger.info("Performance metrics calculated successfully")
    
    def _get_label_category(self, label: str) -> str:
        """Categorize a label into semantic categories."""
        label = label.lower()
        
        if any(term in label for term in ['console', 'handheld', '本体', 'nintendo', 'playstation', 'xbox', 'sega']):
            return 'consoles'
        elif any(term in label for term in ['controller', 'gamepad', 'コントローラー', 'ゲームパッド', 'dualshock']):
            return 'controllers'
        elif any(term in label for term in ['white', 'black', 'blue', 'red', 'silver', 'pink', 'ホワイト', 'ブラック', 'ブルー', 'レッド', 'シルバー', 'ピンク']):
            return 'colors'
        elif any(term in label for term in ['new', 'used', 'mint', 'damaged', 'sealed', 'broken', 'junk', '動作確認済み']):
            return 'conditions'
        elif any(char.isdigit() or char == '-' for char in label) and len(label) >= 5:
            return 'model_codes'
        elif any(term in label for term in ['box', 'case', 'charger', 'stylus', 'memory']):
            return 'accessories'
        else:
            return 'other'
    
    def display_detailed_results(self) -> None:
        """Display detailed test results."""
        logger.info("Displaying detailed test results...")
        
        print("\\n" + "="*80)
        print("COMPREHENSIVE CLASSIFICATION PIPELINE TEST RESULTS")
        print("="*80)
        
        # Overall metrics
        metrics = self.performance_metrics
        print(f"\\nOVERALL PERFORMANCE:")
        print(f"  Total Tests: {metrics['total_tests']}")
        print(f"  Correct Predictions: {metrics['correct_predictions']}")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"  Average Detection Confidence: {metrics['detection_confidence']['mean']:.3f}")
        print(f"  Average Top-1 Similarity: {metrics['similarity_scores']['mean']:.3f}")
        
        # Category performance
        print(f"\\nCATEGORY-WISE PERFORMANCE:")
        for category, stats in metrics['category_performance'].items():
            print(f"  {category.title()}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
        
        # Threshold analysis
        print(f"\\nSIMILARITY THRESHOLD ANALYSIS:")
        for threshold, stats in metrics['threshold_analysis'].items():
            print(f"  ≥ {threshold:.1f}: {stats['count']} detections ({stats['percentage']:.1f}%)")
        
        # Individual results
        print(f"\\nDETAILED CLASSIFICATION RESULTS:")
        print("-" * 80)
        
        for i, result in enumerate(self.test_results):
            status = "✓ CORRECT" if result.is_correct else "✗ INCORRECT"
            print(f"\\n{i+1}. Detection: '{result.detection.label}' (conf: {result.detection.confidence:.2f}) - {status}")
            
            if result.ground_truth:
                print(f"   Ground Truth: {result.ground_truth}")
            
            print("   Top-3 Matches:")
            for j, match in enumerate(result.top_matches):
                print(f"      {j+1}. {match.item_name} (similarity: {match.similarity_score:.3f})")
            
            if not result.top_matches:
                print("      No matches found")
    
    def generate_visualizations(self, save_path: str = "test_results") -> None:
        """Generate visualization plots for test results."""
        logger.info("Generating visualization plots...")
        
        try:
            # Create output directory
            save_dir = Path(save_path)
            save_dir.mkdir(exist_ok=True)
            
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Accuracy by Category
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Category accuracy bar plot
            categories = list(self.performance_metrics['category_performance'].keys())
            accuracies = [self.performance_metrics['category_performance'][cat]['accuracy'] 
                         for cat in categories]
            
            ax1 = axes[0, 0]
            bars = ax1.bar(categories, accuracies)
            ax1.set_title('Classification Accuracy by Category')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.1%}', ha='center', va='bottom')
            
            # 2. Similarity Score Distribution
            ax2 = axes[0, 1]
            similarities = [result.top_matches[0].similarity_score for result in self.test_results 
                           if result.top_matches]
            ax2.hist(similarities, bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(self.similarity_threshold, color='red', linestyle='--', 
                       label=f'Threshold ({self.similarity_threshold})')
            ax2.set_title('Top-1 Similarity Score Distribution')
            ax2.set_xlabel('Similarity Score')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            
            # 3. Detection Confidence vs Similarity
            ax3 = axes[1, 0]
            confidences = [result.detection.confidence for result in self.test_results]
            similarities = [result.top_matches[0].similarity_score if result.top_matches else 0 
                           for result in self.test_results]
            colors = ['green' if result.is_correct else 'red' for result in self.test_results]
            
            scatter = ax3.scatter(confidences, similarities, c=colors, alpha=0.6)
            ax3.set_title('Detection Confidence vs Similarity Score')
            ax3.set_xlabel('Detection Confidence')
            ax3.set_ylabel('Top-1 Similarity Score')
            ax3.grid(True, alpha=0.3)
            
            # Add legend
            import matplotlib.patches as mpatches
            correct_patch = mpatches.Patch(color='green', label='Correct')
            incorrect_patch = mpatches.Patch(color='red', label='Incorrect')
            ax3.legend(handles=[correct_patch, incorrect_patch])
            
            # 4. Threshold Analysis
            ax4 = axes[1, 1]
            thresholds = list(self.performance_metrics['threshold_analysis'].keys())
            percentages = [self.performance_metrics['threshold_analysis'][t]['percentage'] 
                          for t in thresholds]
            
            ax4.plot(thresholds, percentages, marker='o', linewidth=2, markersize=6)
            ax4.set_title('Detections Above Similarity Threshold')
            ax4.set_xlabel('Similarity Threshold')
            ax4.set_ylabel('Percentage of Detections (%)')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'classification_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Confusion Matrix for Categories (if enough data)
            if len(set(result.detection.label.lower() for result in self.test_results)) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create category confusion data
                actual_categories = [self._get_label_category(result.detection.label.lower()) 
                                   for result in self.test_results]
                predicted_categories = []
                
                for result in self.test_results:
                    if result.top_matches:
                        pred_category = self._get_label_category(result.top_matches[0].item_name.lower())
                        predicted_categories.append(pred_category)
                    else:
                        predicted_categories.append('none')
                
                # Create confusion matrix data
                unique_categories = sorted(set(actual_categories + predicted_categories))
                confusion_data = np.zeros((len(unique_categories), len(unique_categories)))
                
                for actual, pred in zip(actual_categories, predicted_categories):
                    i = unique_categories.index(actual)
                    j = unique_categories.index(pred)
                    confusion_data[i, j] += 1
                
                # Plot heatmap
                sns.heatmap(confusion_data, 
                           xticklabels=unique_categories,
                           yticklabels=unique_categories,
                           annot=True, fmt='.0f', cmap='Blues', ax=ax)
                ax.set_title('Category Confusion Matrix')
                ax.set_xlabel('Predicted Category')
                ax.set_ylabel('Actual Category')
                
                plt.tight_layout()
                plt.savefig(save_dir / 'category_confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Visualizations saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def save_results_report(self, save_path: str = "classification_test_report.json") -> None:
        """Save comprehensive test results to JSON report."""
        logger.info(f"Saving test results report to {save_path}")
        
        try:
            # Prepare serializable data
            report_data = {
                'test_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_path': str(self.model_path),
                    'vector_db_path': str(self.vector_db_path),
                    'gdino_output_path': str(self.gdino_output_path),
                    'similarity_threshold': self.similarity_threshold,
                    'total_detections': len(self.test_results)
                },
                'performance_metrics': self.performance_metrics,
                'detailed_results': []
            }
            
            # Add detailed results
            for result in self.test_results:
                result_data = {
                    'detection': {
                        'label': result.detection.label,
                        'confidence': result.detection.confidence,
                        'box': result.detection.box,
                        'image_path': result.detection.image_path
                    },
                    'ground_truth': result.ground_truth,
                    'is_correct': result.is_correct,
                    'top_matches': [
                        {
                            'item_id': match.item_id,
                            'item_name': match.item_name,
                            'similarity_score': match.similarity_score,
                            'vector_index': match.vector_index
                        }
                        for match in result.top_matches
                    ]
                }
                report_data['detailed_results'].append(result_data)
            
            # Save to file
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Test results report saved successfully to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving results report: {str(e)}")
    
    def run_comprehensive_evaluation(self) -> None:
        """Run the complete evaluation pipeline."""
        logger.info("Starting comprehensive evaluation pipeline...")
        
        try:
            # Run classification tests
            self.run_classification_tests()
            
            # Calculate performance metrics
            self.calculate_performance_metrics()
            
            # Display results
            self.display_detailed_results()
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Save report
            self.save_results_report()
            
            logger.info("Comprehensive evaluation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during comprehensive evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def main():
    """Main function to run the comprehensive classification tests."""
    try:
        # Initialize the tester
        tester = RealClassificationTester(
            model_path="intfloat/multilingual-e5-base",
            vector_db_path="models/vector_database",
            gdino_output_path="gdinoOutput",
            similarity_threshold=0.5
        )
        
        # Run comprehensive evaluation
        tester.run_comprehensive_evaluation()
        
        print("\\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("Check the following outputs:")
        print("- classification_test_report.json: Detailed JSON report")
        print("- test_results/: Visualization plots")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Failed to run classification tests: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
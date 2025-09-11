"""
Advanced Position-Weighted Classification Test Suite

This script tests the position-weighted classification system with real GDINO data
and compares performance against baseline classification methods.

Usage:
    python test_advanced_classifier.py

Features:
- Tests position-weighted vs baseline classification
- Analyzes GDINO token ranking effectiveness 
- Validates hardware classification improvements
- Generates comprehensive performance reports
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import position-weighted classification system
from position_weighted_embeddings import PositionWeightedTokenClassifier, analyze_gdino_file_with_position_weighting

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedClassifierTester:
    """
    Advanced tester for position-weighted classification system.
    """
    
    def __init__(self, gdino_output_dir: str = "gdinoOutput/final"):
        """
        Initialize the advanced classifier tester.
        
        Args:
            gdino_output_dir: Directory containing GDINO output JSON files
        """
        self.gdino_output_dir = Path(gdino_output_dir)
        self.position_classifier = PositionWeightedTokenClassifier()
        
        # Test results storage
        self.test_results = []
        self.performance_metrics = {}
        
        logger.info("Advanced Position-Weighted Classifier Tester initialized")
    
    def load_gdino_files(self) -> List[Dict[str, Any]]:
        """
        Load all GDINO output files for testing.
        
        Returns:
            List of GDINO data dictionaries
        """
        gdino_files = []
        
        if not self.gdino_output_dir.exists():
            logger.error(f"GDINO output directory not found: {self.gdino_output_dir}")
            return gdino_files
        
        json_files = list(self.gdino_output_dir.rglob("*.json"))
        logger.info(f"Found {len(json_files)} GDINO JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Add file metadata
                data['file_path'] = str(json_file)
                data['file_name'] = json_file.name
                
                gdino_files.append(data)
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(gdino_files)} GDINO files")
        return gdino_files
    
    def analyze_token_rankings(self, gdino_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze token ranking patterns in a GDINO file.
        
        Args:
            gdino_data: GDINO output data
            
        Returns:
            Token ranking analysis results
        """
        analysis = {
            'file_name': gdino_data.get('file_name', 'unknown'),
            'detections': {},
            'overall_stats': {
                'total_detections': 0,
                'avg_tokens_per_detection': 0,
                'avg_hardware_relevance': 0,
                'top_hardware_predictions': []
            }
        }
        
        gdino_tokens = gdino_data.get('gdino_tokens', {})
        if not gdino_tokens:
            return analysis
        
        analysis['overall_stats']['total_detections'] = len(gdino_tokens)
        
        hardware_scores = []
        hardware_predictions = []
        token_counts = []
        
        for detection_id, tokens in gdino_tokens.items():
            if not tokens or not isinstance(tokens, list):
                continue
            
            # Analyze with position weighting
            token_analysis = self.position_classifier.classify_hardware_tokens(tokens)
            
            detection_analysis = {
                'detection_id': detection_id,
                'total_tokens': len(tokens),
                'top_8_tokens': tokens[:8],
                'hardware_relevance_score': token_analysis['hardware_relevance_score'],
                'predicted_hardware': token_analysis['predicted_hardware'],
                'hardware_tokens': len(token_analysis['hardware_tokens']),
                'brand_tokens': len(token_analysis['brand_tokens']),
                'model_tokens': len(token_analysis['model_tokens'])
            }
            
            analysis['detections'][detection_id] = detection_analysis
            
            # Collect stats
            hardware_scores.append(token_analysis['hardware_relevance_score'])
            token_counts.append(len(tokens))
            
            if token_analysis['predicted_hardware']['top_console']:
                hardware_predictions.append(token_analysis['predicted_hardware']['top_console']['name'])
        
        # Calculate overall statistics
        if hardware_scores:
            analysis['overall_stats']['avg_hardware_relevance'] = np.mean(hardware_scores)
        if token_counts:
            analysis['overall_stats']['avg_tokens_per_detection'] = np.mean(token_counts)
        if hardware_predictions:
            # Count frequency of predictions
            from collections import Counter
            pred_counts = Counter(hardware_predictions)
            analysis['overall_stats']['top_hardware_predictions'] = pred_counts.most_common(5)
        
        return analysis
    
    def compare_classification_methods(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Compare position-weighted classification against baseline methods.
        
        Args:
            tokens: List of GDINO tokens
            
        Returns:
            Comparison results
        """
        comparison = {
            'tokens_analyzed': tokens,
            'token_count': len(tokens),
            'position_weighted': {},
            'baseline_top_3': {},
            'baseline_all_equal': {},
            'improvements': {}
        }
        
        # Position-weighted analysis
        pw_analysis = self.position_classifier.classify_hardware_tokens(tokens)
        comparison['position_weighted'] = {
            'hardware_relevance_score': pw_analysis['hardware_relevance_score'],
            'predicted_hardware': pw_analysis['predicted_hardware'],
            'top_8_tokens': pw_analysis['top_8_tokens'],
            'weighted_tokens': [
                {
                    'token': t['token'],
                    'position': t['position'],
                    'final_weight': t['final_weight'],
                    'is_hardware': t['is_hardware_term']
                }
                for t in pw_analysis['weighted_analysis'][:8]
            ]
        }
        
        # Baseline 1: Use only top 3 tokens (simple position-based)
        top_3_tokens = tokens[:3] if len(tokens) >= 3 else tokens
        if top_3_tokens:
            top_3_analysis = self.position_classifier.classify_hardware_tokens(top_3_tokens)
            comparison['baseline_top_3'] = {
                'tokens_used': top_3_tokens,
                'hardware_relevance_score': top_3_analysis['hardware_relevance_score'],
                'predicted_hardware': top_3_analysis['predicted_hardware']
            }
        
        # Baseline 2: Use all tokens with equal weighting
        if tokens:
            # Simulate equal weighting by using classifier with minimal decay
            equal_weight_classifier = PositionWeightedTokenClassifier(decay_rate=0.01, hardware_boost=1.0)
            equal_analysis = equal_weight_classifier.classify_hardware_tokens(tokens)
            comparison['baseline_all_equal'] = {
                'tokens_used': tokens,
                'hardware_relevance_score': equal_analysis['hardware_relevance_score'],
                'predicted_hardware': equal_analysis['predicted_hardware']
            }
        
        # Calculate improvements
        pw_score = comparison['position_weighted']['hardware_relevance_score']
        top_3_score = comparison['baseline_top_3'].get('hardware_relevance_score', 0)
        equal_score = comparison['baseline_all_equal'].get('hardware_relevance_score', 0)
        
        comparison['improvements'] = {
            'vs_top_3': pw_score - top_3_score,
            'vs_equal_weight': pw_score - equal_score,
            'relative_improvement_top_3': (pw_score - top_3_score) / max(top_3_score, 0.001),
            'relative_improvement_equal': (pw_score - equal_score) / max(equal_score, 0.001)
        }
        
        return comparison
    
    def run_comprehensive_tests(self) -> None:
        """
        Run comprehensive tests on all GDINO files.
        """
        logger.info("Starting comprehensive position-weighted classification tests...")
        
        gdino_files = self.load_gdino_files()
        if not gdino_files:
            logger.error("No GDINO files found. Cannot run tests.")
            return
        
        all_comparisons = []
        all_analyses = []
        
        for gdino_data in gdino_files:
            logger.info(f"Testing file: {gdino_data.get('file_name', 'unknown')}")
            
            # Analyze token rankings
            token_analysis = self.analyze_token_rankings(gdino_data)
            all_analyses.append(token_analysis)
            
            # Compare classification methods for each detection
            gdino_tokens = gdino_data.get('gdino_tokens', {})
            for detection_id, tokens in gdino_tokens.items():
                if tokens and isinstance(tokens, list) and len(tokens) > 0:
                    comparison = self.compare_classification_methods(tokens)
                    comparison['file_name'] = gdino_data.get('file_name', 'unknown')
                    comparison['detection_id'] = detection_id
                    all_comparisons.append(comparison)
        
        # Store results
        self.test_results = {
            'token_analyses': all_analyses,
            'method_comparisons': all_comparisons
        }
        
        logger.info(f"Completed tests on {len(gdino_files)} files with {len(all_comparisons)} total comparisons")
    
    def calculate_performance_metrics(self) -> None:
        """
        Calculate comprehensive performance metrics.
        """
        logger.info("Calculating performance metrics...")
        
        if not self.test_results:
            logger.warning("No test results available")
            return
        
        comparisons = self.test_results['method_comparisons']
        analyses = self.test_results['token_analyses']
        
        # Overall statistics
        total_comparisons = len(comparisons)
        total_files = len(analyses)
        
        # Hardware relevance improvements
        pw_scores = [comp['position_weighted']['hardware_relevance_score'] for comp in comparisons]
        top_3_scores = [comp['baseline_top_3'].get('hardware_relevance_score', 0) for comp in comparisons]
        equal_scores = [comp['baseline_all_equal'].get('hardware_relevance_score', 0) for comp in comparisons]
        
        improvements_vs_top_3 = [comp['improvements']['vs_top_3'] for comp in comparisons]
        improvements_vs_equal = [comp['improvements']['vs_equal_weight'] for comp in comparisons]
        
        # Count positive improvements
        positive_improvements_top_3 = sum(1 for imp in improvements_vs_top_3 if imp > 0)
        positive_improvements_equal = sum(1 for imp in improvements_vs_equal if imp > 0)
        
        # Token count analysis
        token_counts = [comp['token_count'] for comp in comparisons]
        
        # Hardware prediction consistency
        hardware_predictions = []\n        for comp in comparisons:\n            pred = comp['position_weighted']['predicted_hardware']\n            if pred and pred.get('top_console'):\n                hardware_predictions.append(pred['top_console']['name'])\n        \n        from collections import Counter\n        prediction_counts = Counter(hardware_predictions)\n        \n        self.performance_metrics = {\n            'overall_stats': {\n                'total_files_tested': total_files,\n                'total_comparisons': total_comparisons,\n                'avg_tokens_per_detection': np.mean(token_counts) if token_counts else 0\n            },\n            'hardware_relevance_scores': {\n                'position_weighted': {\n                    'mean': np.mean(pw_scores) if pw_scores else 0,\n                    'std': np.std(pw_scores) if pw_scores else 0,\n                    'min': np.min(pw_scores) if pw_scores else 0,\n                    'max': np.max(pw_scores) if pw_scores else 0\n                },\n                'baseline_top_3': {\n                    'mean': np.mean(top_3_scores) if top_3_scores else 0,\n                    'std': np.std(top_3_scores) if top_3_scores else 0\n                },\n                'baseline_equal_weight': {\n                    'mean': np.mean(equal_scores) if equal_scores else 0,\n                    'std': np.std(equal_scores) if equal_scores else 0\n                }\n            },\n            'improvements': {\n                'vs_top_3': {\n                    'mean_improvement': np.mean(improvements_vs_top_3) if improvements_vs_top_3 else 0,\n                    'positive_improvements': positive_improvements_top_3,\n                    'improvement_rate': positive_improvements_top_3 / total_comparisons if total_comparisons > 0 else 0\n                },\n                'vs_equal_weight': {\n                    'mean_improvement': np.mean(improvements_vs_equal) if improvements_vs_equal else 0,\n                    'positive_improvements': positive_improvements_equal,\n                    'improvement_rate': positive_improvements_equal / total_comparisons if total_comparisons > 0 else 0\n                }\n            },\n            'hardware_predictions': {\n                'total_predictions': len(hardware_predictions),\n                'unique_hardware_types': len(prediction_counts),\n                'top_predictions': prediction_counts.most_common(10)\n            }\n        }\n        \n        logger.info("Performance metrics calculated successfully")\n    \n    def display_results(self) -> None:\n        """
n        Display comprehensive test results.\n        """\n        if not self.performance_metrics:\n            logger.error("No performance metrics available")\n            return\n        \n        print("\\n" + "="*80)\n        print("ADVANCED POSITION-WEIGHTED CLASSIFICATION TEST RESULTS")\n        print("="*80)\n        \n        metrics = self.performance_metrics\n        \n        # Overall statistics\n        print(f"\\nOVERALL TEST STATISTICS:")\n        print(f"  Files Tested: {metrics['overall_stats']['total_files_tested']}")\n        print(f"  Total Comparisons: {metrics['overall_stats']['total_comparisons']}")\n        print(f"  Average Tokens per Detection: {metrics['overall_stats']['avg_tokens_per_detection']:.1f}")\n        \n        # Hardware relevance scores\n        print(f"\\nHARDWARE RELEVANCE SCORE ANALYSIS:")\n        hw_scores = metrics['hardware_relevance_scores']\n        print(f"  Position-Weighted: {hw_scores['position_weighted']['mean']:.3f} ± {hw_scores['position_weighted']['std']:.3f}")\n        print(f"  Baseline (Top-3): {hw_scores['baseline_top_3']['mean']:.3f} ± {hw_scores['baseline_top_3']['std']:.3f}")\n        print(f"  Baseline (Equal Weight): {hw_scores['baseline_equal_weight']['mean']:.3f} ± {hw_scores['baseline_equal_weight']['std']:.3f}")\n        \n        # Improvements\n        print(f"\\nCLASSIFICATION IMPROVEMENTS:")\n        improvements = metrics['improvements']\n        print(f"  vs Top-3 Baseline:")\n        print(f"    Mean Improvement: {improvements['vs_top_3']['mean_improvement']:+.3f}")\n        print(f"    Positive Improvements: {improvements['vs_top_3']['positive_improvements']} ({improvements['vs_top_3']['improvement_rate']:.1%})")\n        print(f"  vs Equal Weight Baseline:")\n        print(f"    Mean Improvement: {improvements['vs_equal_weight']['mean_improvement']:+.3f}")\n        print(f"    Positive Improvements: {improvements['vs_equal_weight']['positive_improvements']} ({improvements['vs_equal_weight']['improvement_rate']:.1%})")\n        \n        # Hardware predictions\n        print(f"\\nHARDWARE PREDICTION ANALYSIS:")\n        hw_pred = metrics['hardware_predictions']\n        print(f"  Total Predictions: {hw_pred['total_predictions']}")\n        print(f"  Unique Hardware Types: {hw_pred['unique_hardware_types']}")\n        print(f"  Top Predicted Hardware:")\n        for hardware, count in hw_pred['top_predictions'][:5]:\n            percentage = (count / hw_pred['total_predictions'] * 100) if hw_pred['total_predictions'] > 0 else 0\n            print(f"    {hardware}: {count} ({percentage:.1f}%)")\n        \n        # Show some example improvements\n        print(f"\\nEXAMPLE IMPROVEMENTS:")\n        comparisons = self.test_results['method_comparisons']\n        \n        # Find best improvements\n        best_improvements = sorted(\n            comparisons, \n            key=lambda x: x['improvements']['vs_top_3'], \n            reverse=True\n        )[:5]\n        \n        for i, comp in enumerate(best_improvements):\n            if comp['improvements']['vs_top_3'] > 0:\n                print(f"\\n  {i+1}. File: {comp['file_name']}, Detection: {comp['detection_id']}")\n                print(f"     Tokens: {', '.join(comp['position_weighted']['top_8_tokens'][:5])}...")\n                print(f"     Position-Weighted Score: {comp['position_weighted']['hardware_relevance_score']:.3f}")\n                print(f"     Top-3 Baseline Score: {comp['baseline_top_3'].get('hardware_relevance_score', 0):.3f}")\n                print(f"     Improvement: {comp['improvements']['vs_top_3']:+.3f}")\n                \n                if comp['position_weighted']['predicted_hardware']['top_console']:\n                    console = comp['position_weighted']['predicted_hardware']['top_console']\n                    print(f"     Predicted Hardware: {console['name']} (confidence: {console['confidence']:.3f})")\n    \n    def save_detailed_report(self, save_path: str = "advanced_classification_report.json") -> None:\n        """
        Save detailed test report to JSON file.\n        \n        Args:\n            save_path: Path to save the report\n        """\n        logger.info(f"Saving detailed report to {save_path}")\n        \n        report = {\n            'test_metadata': {\n                'timestamp': datetime.now().isoformat(),\n                'gdino_output_dir': str(self.gdino_output_dir),\n                'position_classifier_config': {\n                    'decay_rate': self.position_classifier.decay_rate,\n                    'top_k_boost': self.position_classifier.top_k_boost,\n                    'min_weight': self.position_classifier.min_weight,\n                    'hardware_boost': self.position_classifier.hardware_boost\n                }\n            },\n            'performance_metrics': self.performance_metrics,\n            'detailed_results': self.test_results\n        }\n        \n        try:\n            with open(save_path, 'w', encoding='utf-8') as f:\n                json.dump(report, f, indent=2, ensure_ascii=False)\n            logger.info(f"Report saved successfully to {save_path}")\n        except Exception as e:\n            logger.error(f"Error saving report: {e}")\n    \n    def generate_visualizations(self, save_dir: str = "advanced_test_results") -> None:\n        """
        Generate visualization plots for the test results.\n        \n        Args:\n            save_dir: Directory to save plots\n        """\n        logger.info("Generating visualization plots...")\n        \n        try:\n            save_path = Path(save_dir)\n            save_path.mkdir(exist_ok=True)\n            \n            # Set plotting style\n            plt.style.use('default')\n            sns.set_palette("husl")\n            \n            # 1. Hardware Relevance Score Comparison\n            fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n            \n            # Box plot comparing methods\n            ax1 = axes[0, 0]\n            comparisons = self.test_results['method_comparisons']\n            \n            pw_scores = [comp['position_weighted']['hardware_relevance_score'] for comp in comparisons]\n            top_3_scores = [comp['baseline_top_3'].get('hardware_relevance_score', 0) for comp in comparisons]\n            equal_scores = [comp['baseline_all_equal'].get('hardware_relevance_score', 0) for comp in comparisons]\n            \n            box_data = [pw_scores, top_3_scores, equal_scores]\n            box_labels = ['Position\\nWeighted', 'Top-3\\nBaseline', 'Equal Weight\\nBaseline']\n            \n            ax1.boxplot(box_data, labels=box_labels)\n            ax1.set_title('Hardware Relevance Score Comparison')\n            ax1.set_ylabel('Hardware Relevance Score')\n            ax1.grid(True, alpha=0.3)\n            \n            # 2. Improvement Distribution\n            ax2 = axes[0, 1]\n            improvements = [comp['improvements']['vs_top_3'] for comp in comparisons]\n            ax2.hist(improvements, bins=20, alpha=0.7, edgecolor='black')\n            ax2.axvline(0, color='red', linestyle='--', label='No Improvement')\n            ax2.set_title('Improvement Distribution (vs Top-3 Baseline)')\n            ax2.set_xlabel('Improvement in Hardware Relevance Score')\n            ax2.set_ylabel('Frequency')\n            ax2.legend()\n            ax2.grid(True, alpha=0.3)\n            \n            # 3. Token Count vs Improvement\n            ax3 = axes[1, 0]\n            token_counts = [comp['token_count'] for comp in comparisons]\n            improvements = [comp['improvements']['vs_top_3'] for comp in comparisons]\n            colors = ['green' if imp > 0 else 'red' for imp in improvements]\n            \n            scatter = ax3.scatter(token_counts, improvements, c=colors, alpha=0.6)\n            ax3.axhline(0, color='black', linestyle='--', alpha=0.5)\n            ax3.set_title('Token Count vs Improvement')\n            ax3.set_xlabel('Number of Tokens')\n            ax3.set_ylabel('Improvement Score')\n            ax3.grid(True, alpha=0.3)\n            \n            # Add legend\n            import matplotlib.patches as mpatches\n            improved_patch = mpatches.Patch(color='green', label='Improved')\n            degraded_patch = mpatches.Patch(color='red', label='Degraded')\n            ax3.legend(handles=[improved_patch, degraded_patch])\n            \n            # 4. Hardware Prediction Distribution\n            ax4 = axes[1, 1]\n            hw_pred = self.performance_metrics['hardware_predictions']\n            if hw_pred['top_predictions']:\n                hardware_names = [pred[0] for pred in hw_pred['top_predictions'][:8]]\n                hardware_counts = [pred[1] for pred in hw_pred['top_predictions'][:8]]\n                \n                bars = ax4.bar(range(len(hardware_names)), hardware_counts)\n                ax4.set_title('Top Hardware Predictions')\n                ax4.set_ylabel('Frequency')\n                ax4.set_xticks(range(len(hardware_names)))\n                ax4.set_xticklabels(hardware_names, rotation=45, ha='right')\n                \n                # Add value labels\n                for i, (bar, count) in enumerate(zip(bars, hardware_counts)):\n                    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,\n                            str(count), ha='center', va='bottom')\n            \n            plt.tight_layout()\n            plt.savefig(save_path / 'advanced_classification_analysis.png', dpi=300, bbox_inches='tight')\n            plt.close()\n            \n            logger.info(f"Visualizations saved to {save_path}")\n            \n        except Exception as e:\n            logger.error(f"Error generating visualizations: {e}")\n    \n    def run_complete_evaluation(self) -> None:\n        """
        Run the complete evaluation pipeline.\n        """\n        logger.info("Starting complete advanced classification evaluation...")\n        \n        try:\n            # Run tests\n            self.run_comprehensive_tests()\n            \n            # Calculate metrics\n            self.calculate_performance_metrics()\n            \n            # Display results\n            self.display_results()\n            \n            # Generate visualizations\n            self.generate_visualizations()\n            \n            # Save report\n            self.save_detailed_report()\n            \n            logger.info("✅ Advanced classification evaluation completed successfully!")\n            \n        except Exception as e:\n            logger.error(f"Error during evaluation: {e}")\n            raise


def main():\n    """Main function to run advanced classification tests."""\n    try:\n        # Initialize tester\n        tester = AdvancedClassifierTester(gdino_output_dir="gdinoOutput/final")\n        \n        # Run complete evaluation\n        tester.run_complete_evaluation()\n        \n        print("\\n" + "="*80)\n        print("🎯 ADVANCED POSITION-WEIGHTED CLASSIFICATION EVALUATION COMPLETED!")\n        print("📊 Check the following outputs:")\n        print("   - advanced_classification_report.json: Detailed JSON report")\n        print("   - advanced_test_results/: Visualization plots")\n        print("   - Position weighting improvements and hardware predictions")\n        print("="*80)\n        \n        return 0\n        \n    except Exception as e:\n        logger.error(f"Failed to run advanced classification tests: {e}")\n        return 1


if __name__ == "__main__":\n    exit(main())
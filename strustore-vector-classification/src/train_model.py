"""
Semantic Model Training Script for Gaming Console Classification

This script loads training data from CSV files, fine-tunes a multilingual
sentence-transformer model, and saves the result for vector-based classification.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

# Core ML libraries
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from torch.utils.data import DataLoader
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GamingConsoleSemanticTrainer:
    """
    Trainer class for fine-tuning semantic embeddings for gaming console classification.
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.training_data = []
        self.evaluation_data = []
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.model_output_path = Path(self.config['model_output_path'])
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration optimized for small datasets."""
        return {
            'base_model_name': 'paraphrase-multilingual-mpnet-base-v2',
            'positive_pairs_path': 'src/data/training_data/positive_pairs.csv',
            'triplets_path': 'src/data/training_data/triplets.csv',
            'model_output_path': 'models/gaming-console-semantic-model',
            'batch_size': 16,  # Smaller batch size for better learning
            'num_epochs': 15,  # More epochs to learn from small data
            'warmup_steps_ratio': 0.2,  # More warmup for stable learning
            'evaluation_steps': 50,  # More frequent evaluation
            'save_best_model': True,
            'max_seq_length': 384,
            'learning_rate': 2e-5,  # Lower learning rate for fine-tuning
            'weight_decay': 0.01
        }
    
    def load_base_model(self) -> None:
        """Load the pre-trained multilingual sentence transformer model."""
        try:
            logger.info(f"Loading base model: {self.config['base_model_name']}")
            self.model = SentenceTransformer(
                self.config['base_model_name'],
                device=self.device
            )
            
            # Set max sequence length
            self.model.max_seq_length = self.config['max_seq_length']
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load base model: {e}")
    
    def load_training_data(self) -> None:
        """Load and prepare training data with triplet generation from existing pairs."""
        try:
            # Load positive pairs
            positive_pairs_path = Path(self.config['positive_pairs_path'])
            if not positive_pairs_path.exists():
                raise FileNotFoundError(f"Positive pairs file not found: {positive_pairs_path}")
            
            # Read CSV and filter out comment lines
            df = pd.read_csv(positive_pairs_path)
            df = df[~df['item1'].str.startswith('#', na=False)]  # Remove comment lines
            df = df.dropna()  # Remove any NaN rows
            
            logger.info(f"Loaded {len(df)} positive pairs from {positive_pairs_path}")
            
            # Extract all unique terms from your existing data
            all_terms = set()
            positive_pairs = []
            
            for _, row in df.iterrows():
                item1 = str(row['item1']).strip()
                item2 = str(row['item2']).strip()
                
                # Skip empty or invalid entries
                if not item1 or not item2 or item1 == item2:
                    continue
                
                positive_pairs.append((item1, item2))
                all_terms.add(item1)
                all_terms.add(item2)
            
            logger.info(f"Found {len(all_terms)} unique terms in your data")
            
            # Create semantic categories from your existing data
            categories = {
                'colors': set(),
                'consoles': set(), 
                'conditions': set(),
                'brands': set(),
                'model_codes': set(),
                'synonyms': set()
            }
            
            # Categorize your existing terms
            for term in all_terms:
                term_lower = term.lower()
                if any(color in term_lower for color in ['white', 'black', 'red', 'blue', 'pink', 'ice', 'azul', 'negro', 'rosa', 'ホワイト', 'ブラック']):
                    categories['colors'].add(term)
                elif any(console in term_lower for console in ['ds', 'gameboy', 'playstation', 'nintendo', 'famicom', 'console', '本体', 'プレイステーション', '任天堂']):
                    categories['consoles'].add(term)
                elif any(cond in term_lower for cond in ['new', 'used', 'tested', 'pre-owned', 'working', 'cib']):
                    categories['conditions'].add(term)
                elif any(brand in term_lower for brand in ['nintendo', 'sony', 'playstation', 'snes', 'ps']):
                    categories['brands'].add(term)
                elif any(code in term for code in ['NTR-001', 'USG-001', 'TWL-001', 'AGS-001', 'SHVC-001']):
                    categories['model_codes'].add(term)
                else:
                    categories['synonyms'].add(term)
            
            # Generate valid triplets from your existing data
            triplets = []
            
            # For each positive pair, create triplets with negatives from different categories
            for item1, item2 in positive_pairs:
                # Find which category item1 belongs to
                item1_category = None
                for cat_name, cat_terms in categories.items():
                    if item1 in cat_terms:
                        item1_category = cat_name
                        break
                
                if item1_category:
                    # Create negatives from other categories
                    for cat_name, cat_terms in categories.items():
                        if cat_name != item1_category and cat_terms:
                            # Pick a negative from a different category
                            negative = next(iter(cat_terms))
                            triplets.append((item1, item2, negative))  # (anchor, positive, negative)
                            triplets.append((item2, item1, negative))  # Bidirectional
            
            # Also create challenging negatives within similar categories
            console_terms = categories['consoles']
            if len(console_terms) >= 3:
                console_list = list(console_terms)
                for i, anchor in enumerate(console_list):
                    for j, positive in enumerate(console_list):
                        if i != j:
                            # Find a positive pair that matches
                            if any((anchor, positive) == pair or (positive, anchor) == pair for pair in positive_pairs):
                                # Use another console as negative  
                                for k, negative in enumerate(console_list):
                                    if k != i and k != j:
                                        triplets.append((anchor, positive, negative))
            
            logger.info(f"Generated {len(triplets)} valid triplets from your existing data")
            
            # Convert to InputExample format for sentence-transformers
            from sklearn.model_selection import train_test_split
            
            # Random split for better evaluation
            if len(triplets) > 0:
                train_triplets, eval_triplets = train_test_split(
                    triplets, test_size=0.15, random_state=42, shuffle=True
                )
            else:
                # Fallback to positive pairs if no triplets generated
                train_triplets, eval_triplets = train_test_split(
                    [(p[0], p[1], "random_negative") for p in positive_pairs], 
                    test_size=0.15, random_state=42, shuffle=True
                )
            
            # Create InputExamples for triplet training
            self.training_data = []
            self.evaluation_data = []
            
            for anchor, positive, negative in train_triplets:
                # Create triplet example: anchor should be closer to positive than negative
                example = InputExample(texts=[anchor, positive, negative])
                self.training_data.append(example)
            
            for anchor, positive, negative in eval_triplets:
                example = InputExample(texts=[anchor, positive, negative])
                self.evaluation_data.append(example)
            
            logger.info(f"Prepared {len(self.training_data)} triplet training examples")
            logger.info(f"Prepared {len(self.evaluation_data)} triplet evaluation examples")
            
            if len(self.training_data) == 0:
                raise ValueError("No valid training data found")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load training data: {e}")
    
    def create_data_loader(self) -> DataLoader:
        """Create DataLoader for training."""
        return DataLoader(
            self.training_data,
            shuffle=True,
            batch_size=self.config['batch_size']
        )
    
    def setup_loss_function(self):
        """Setup the loss function for triplet training."""
        # TripletLoss is optimal for learning fine-grained relationships
        # It learns that anchor should be closer to positive than to negative
        return losses.TripletLoss(self.model, triplet_margin=0.5)
    
    def setup_evaluator(self):
        """Setup evaluator for monitoring training progress."""
        if len(self.evaluation_data) == 0:
            return None
            
        # Create evaluation pairs
        sentences1 = [example.texts[0] for example in self.evaluation_data]
        sentences2 = [example.texts[1] for example in self.evaluation_data]
        scores = [example.label for example in self.evaluation_data]
        
        # Use cosine similarity evaluator
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            sentences1, sentences2, scores,
            name='cosine_similarity_evaluation'
        )
        
        return evaluator
    
    def train(self) -> None:
        """Execute the training process."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_base_model() first.")
        
        if not self.training_data:
            raise RuntimeError("Training data not loaded. Call load_training_data() first.")
        
        logger.info("=== Starting Semantic Model Training ===")
        
        # Create data loader
        train_dataloader = self.create_data_loader()
        
        # Setup loss function
        train_loss = self.setup_loss_function()
        
        # Setup evaluator
        evaluator = self.setup_evaluator()
        
        # Calculate warmup steps
        num_training_steps = len(train_dataloader) * self.config['num_epochs']
        warmup_steps = int(num_training_steps * self.config['warmup_steps_ratio'])
        
        logger.info(f"Training configuration:")
        logger.info(f"  - Epochs: {self.config['num_epochs']}")
        logger.info(f"  - Batch size: {self.config['batch_size']}")
        logger.info(f"  - Training steps: {num_training_steps}")
        logger.info(f"  - Warmup steps: {warmup_steps}")
        logger.info(f"  - Loss function: TripletLoss (margin=0.5)")
        
        # Training arguments
        training_args = {
            'train_objectives': [(train_dataloader, train_loss)],
            'epochs': self.config['num_epochs'],
            'warmup_steps': warmup_steps,
            'output_path': str(self.model_output_path),
            'show_progress_bar': True,
            'save_best_model': self.config['save_best_model']
        }
        
        # Add evaluator if available
        if evaluator is not None:
            training_args['evaluator'] = evaluator
            training_args['evaluation_steps'] = self.config['evaluation_steps']
        
        # Start training
        start_time = datetime.now()
        logger.info(f"Training started at: {start_time}")
        
        try:
            self.model.fit(**training_args)
            
            end_time = datetime.now()
            training_duration = end_time - start_time
            
            logger.info(f"Training completed at: {end_time}")
            logger.info(f"Training duration: {training_duration}")
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")
    
    def save_model_info(self) -> None:
        """Save model information and training metadata."""
        try:
            model_info = {
                'model_name': 'gaming-console-semantic-model',
                'base_model': self.config['base_model_name'],
                'training_date': datetime.now().isoformat(),
                'training_config': self.config,
                'model_dimension': self.model.get_sentence_embedding_dimension(),
                'training_examples_count': len(self.training_data),
                'evaluation_examples_count': len(self.evaluation_data),
                'device_used': str(self.device),
                'pytorch_version': torch.__version__
            }
            
            # Save to JSON file
            info_path = self.model_output_path / 'model_info.json'
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Model information saved to: {info_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model info: {e}")
    
    def test_model_basic(self) -> None:
        """Test the trained model with triplet relationships."""
        if self.model is None:
            logger.warning("No model available for testing")
            return
        
        try:
            logger.info("=== Testing Trained Model with Triplet Relationships ===")
            
            # Test triplets - anchor should be closer to positive than negative
            test_triplets = [
                ("console", "本体", "コントローラー"),  # console closer to 本体 than controller
                ("white", "ホワイト", "PlayStation"),   # white closer to ホワイト than PlayStation
                ("Nintendo DS Lite Console", "USG-001", "TWL-001"),  # DS Lite closer to USG-001 than TWL-001
                ("controller", "gamepad", "本体"),      # controller closer to gamepad than console
            ]
            
            logger.info("Triplet relationship scores (anchor -> positive vs negative):")
            for anchor, positive, negative in test_triplets:
                embeddings = self.model.encode([anchor, positive, negative])
                
                # Calculate similarities
                anchor_pos_sim = self.model.similarity([embeddings[0]], [embeddings[1]]).numpy()[0][0]
                anchor_neg_sim = self.model.similarity([embeddings[0]], [embeddings[2]]).numpy()[0][0]
                
                triplet_success = anchor_pos_sim > anchor_neg_sim
                status = "✅" if triplet_success else "❌"
                
                logger.info(f"  {status} '{anchor}' -> '{positive}': {anchor_pos_sim:.4f} vs '{negative}': {anchor_neg_sim:.4f}")
            
            # Test some positive pairs for absolute similarity
            logger.info("\nCore positive pair similarities:")
            test_pairs = [
                ("white", "ホワイト"),
                ("console", "本体"),
                ("nintendo", "任天堂")
            ]
            
            for item1, item2 in test_pairs:
                embeddings = self.model.encode([item1, item2])
                similarity = self.model.similarity(embeddings, embeddings).numpy()
                score = similarity[0][1]
                logger.info(f"  '{item1}' <-> '{item2}': {score:.4f}")
                
        except Exception as e:
            logger.error(f"Model testing failed: {e}")
    
    def run_full_training_pipeline(self) -> None:
        """Execute the complete training pipeline."""
        try:
            logger.info("🚀 Starting Gaming Console Semantic Training Pipeline")
            
            # Step 1: Load base model
            logger.info("\n📦 Step 1: Loading Base Model")
            self.load_base_model()
            
            # Step 2: Load training data
            logger.info("\n📊 Step 2: Loading Training Data")
            self.load_training_data()
            
            # Step 3: Train model
            logger.info("\n🏋️ Step 3: Training Model")
            self.train()
            
            # Step 4: Save model information
            logger.info("\n💾 Step 4: Saving Model Information")
            self.save_model_info()
            
            # Step 5: Test model
            logger.info("\n🧪 Step 5: Testing Trained Model")
            self.test_model_basic()
            
            logger.info("\n✅ Training Pipeline Completed Successfully!")
            logger.info(f"📁 Model saved to: {self.model_output_path}")
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            raise


def main():
    """Main function to run the training."""
    
    # Custom configuration (optional)
    config = {
        'base_model_name': 'paraphrase-multilingual-mpnet-base-v2',
        'positive_pairs_path': 'src/data/training_data/positive_pairs.csv',
        'model_output_path': 'models/gaming-console-semantic-model',
        'batch_size': 32,
        'num_epochs': 5,
        'warmup_steps_ratio': 0.1,
        'evaluation_steps': 500,
        'save_best_model': True,
        'max_seq_length': 384
    }
    
    # Initialize trainer
    trainer = GamingConsoleSemanticTrainer(config)
    
    # Run training pipeline
    trainer.run_full_training_pipeline()


if __name__ == '__main__':
    main()
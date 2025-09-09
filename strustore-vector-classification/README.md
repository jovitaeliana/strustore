# Gaming Console Semantic Model Training

This system implements semantic embedding fine-tuning for gaming console classification in auction listings. It trains a multilingual sentence-transformer model to understand relationships between Japanese and English gaming terms, console names, and technical specifications.

## 🎯 Overview

The system consists of three main components:

1. **Training Pipeline** (`train_model.py`) - Fine-tunes a multilingual sentence-transformer model
2. **Evaluation Pipeline** (`evaluate_model.py`) - Comprehensive model quality assurance  
3. **Vector Database Creation** (`create_vector_database.py`) - Creates searchable vector database for real-time classification

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- 8GB+ RAM (16GB recommended for training)
- GPU recommended but not required

### Data Requirements
- `positive_pairs.csv` - Training data with semantic pairs
- Master items CSV file (e.g., `items.csv`) - Canonical item list for vector database

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Train the Semantic Model

```bash
python src/train_model.py
```

This will:
- Load a pre-trained multilingual model (`paraphrase-multilingual-mpnet-base-v2`)
- Fine-tune it on your gaming console semantic pairs
- Save the trained model to `models/gaming-console-semantic-model/`

**Expected Output:**
```
🚀 Starting Gaming Console Semantic Training Pipeline
📦 Step 1: Loading Base Model
📊 Step 2: Loading Training Data
🏋️ Step 3: Training Model
💾 Step 4: Saving Model Information
🧪 Step 5: Testing Trained Model
✅ Training Pipeline Completed Successfully!
```

### 3. Evaluate Model Quality

```bash
python src/evaluate_model.py
```

This runs comprehensive tests:
- **Semantic Unit Tests** - Verifies specific learned relationships
- **Negative Similarity Tests** - Ensures dissimilar items have low similarity
- **Console Classification Tests** - Tests classification accuracy
- **Embedding Quality Analysis** - Analyzes embedding properties

**Target:** >85% overall accuracy for deployment approval.

### 4. Create Vector Database

```bash
python src/create_vector_database.py
```

This creates the production vector database:
- Generates embeddings for all master items
- Creates FAISS index for fast similarity search
- Saves complete database to `models/vector_database/`

## 📁 Project Structure

```
strustore-vector-classification/
├── src/
│   ├── data/
│   │   └── training_data/
│   │       ├── positive_pairs.csv    # Training data
│   │       └── triplets.csv         # Advanced training data
│   ├── train_model.py               # Main training script
│   ├── evaluate_model.py            # Model evaluation
│   └── create_vector_database.py    # Vector DB creation
├── models/
│   ├── gaming-console-semantic-model/  # Trained model
│   └── vector_database/                # Production database
├── requirements.txt                 # Dependencies
└── README.md                       # This file
```

## 🔧 Configuration

### Training Configuration

Edit `train_model.py` to customize training:

```python
config = {
    'base_model_name': 'paraphrase-multilingual-mpnet-base-v2',
    'positive_pairs_path': 'src/data/training_data/positive_pairs.csv',
    'model_output_path': 'models/gaming-console-semantic-model',
    'batch_size': 32,
    'num_epochs': 5,
    'warmup_steps_ratio': 0.1,
    'max_seq_length': 384
}
```

### Evaluation Thresholds

Modify `evaluate_model.py` for different similarity thresholds:

```python
evaluator = SemanticModelEvaluator(
    model_path="models/gaming-console-semantic-model",
    threshold=0.75  # Similarity threshold for positive predictions
)
```

## 📊 Training Data Format

### positive_pairs.csv

```csv
item1,item2
Nintendo DS,ニンテンドーDS
console,本体
controller,コントローラー
white,ホワイト
used,中古
Game Boy Advance,GBA
PlayStation,PS
complete in box,cib
```

**Categories covered:**
- **Core Translations** - Japanese ↔ English terms
- **Console Names** - Brand names and abbreviations
- **Colors** - Color terminology in multiple languages  
- **Conditions** - Item condition terms
- **Model Codes** - Technical model identifiers
- **Synonyms** - Alternative names and abbreviations

## 🎯 Model Performance Targets

| Test Category | Target Accuracy | Purpose |
|---------------|----------------|---------|
| Semantic Unit Tests | >85% | Core translation pairs work correctly |
| Negative Similarity | >80% | Model distinguishes dissimilar items |
| Console Classification | >85% | Accurate console identification |
| **Overall Score** | **>85%** | **Deployment approval threshold** |

## 📈 Usage in Production

### Loading the Trained Model

```python
from sentence_transformers import SentenceTransformer

# Load trained model
model = SentenceTransformer('models/gaming-console-semantic-model')

# Generate embeddings
text = "Nintendo DS Lite"
embedding = model.encode([text])
```

### Using Vector Database

```python
from create_vector_database import VectorDatabaseLoader

# Load database
db = VectorDatabaseLoader(
    database_path="models/vector_database",
    model_path="models/gaming-console-semantic-model"
)
db.load_database()

# Search for similar items
results = db.search("ニンテンドーDS", k=5)
for result in results:
    print(f"ID: {result['id']}, Name: {result['name']}, Score: {result['similarity_score']:.3f}")
```

## 🔄 Integration with GroundingDINO Pipeline

### Real-time Classification Flow:

1. **GroundingDINO** extracts text label from auction image (e.g., `"ニンテンドーds"`)
2. **Semantic Model** converts text to 768-dimensional vector
3. **Vector Database** finds most similar item using cosine similarity
4. **Result** returns structured classification: `{"product_id": 2, "product_name": "Nintendo DS Lite Console"}`

### Example Integration:

```python
# 1. Load components
model = SentenceTransformer('models/gaming-console-semantic-model')
db = VectorDatabaseLoader('models/vector_database', 'models/gaming-console-semantic-model')
db.load_database()

# 2. Process GroundingDINO output
grounding_dino_text = "ニンテンドーds"  # From image detection

# 3. Classify
results = db.search(grounding_dino_text, k=1)
best_match = results[0]

# 4. Return structured result
classification = {
    "product_id": best_match['id'], 
    "product_name": best_match['name'],
    "confidence_score": best_match['similarity_score']
}
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   ```bash
   # Install CPU-only version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Memory Issues During Training**
   - Reduce `batch_size` in config
   - Use smaller base model: `'all-MiniLM-L6-v2'`

3. **Low Evaluation Scores**
   - Add more training pairs to `positive_pairs.csv`
   - Increase training epochs
   - Check data quality

4. **Model Not Found Errors**
   - Run `train_model.py` before `evaluate_model.py`
   - Check file paths in configuration

### Performance Optimization

**Training Speed:**
- Use GPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Increase batch size if memory allows
- Use smaller evaluation intervals

**Search Speed:**
- FAISS GPU index: `pip install faiss-gpu`
- Optimize FAISS index parameters for your dataset size

## 📝 Model Information

### Base Model
- **Name:** `paraphrase-multilingual-mpnet-base-v2`
- **Dimensions:** 768
- **Languages:** 50+ including Japanese, English
- **Provider:** sentence-transformers

### Training Details
- **Loss Function:** MultipleNegativesRankingLoss
- **Optimizer:** AdamW with warmup
- **Training Data:** Gaming console semantic pairs
- **Validation:** 20% split for evaluation

## 🤝 Contributing

To add new training data:

1. Edit `src/data/training_data/positive_pairs.csv`
2. Add semantic pairs in format: `item1,item2`
3. Retrain model: `python src/train_model.py`
4. Validate: `python src/evaluate_model.py`
5. Update database: `python src/create_vector_database.py`

## 📄 License

This project is part of the strustore auction classification system.
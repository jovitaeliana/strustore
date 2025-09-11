# Strustore Vector Classification Database Commands Guide

## Quick Navigation
- [Database Status Commands](#database-status-commands)
- [Testing Commands](#testing-commands)
- [Position-Weighted Classification](#position-weighted-classification)
- [Codebase Overview](#codebase-overview)
- [Vector Embedding Explanation](#vector-embedding-explanation)
- [File Structure Analysis](#file-structure-analysis)

---

## Database Status Commands

### Navigate to Project Directory from strustore home
```bash
# If you're starting from: (strustore_env) jovitaeliana@Jo strustore %
cd strustore-vector-classification

# Or from anywhere else:
cd /Users/jovitaeliana/Personal/strustore/strustore-vector-classification
```

### Check Database Health
```bash
# From strustore-vector-classification directory
python -c "
import faiss
import json
import numpy as np

# Check FAISS index
index = faiss.read_index('models/vector_database/faiss_index.bin')
print(f'✅ FAISS Index: {index.ntotal} vectors, {index.d} dimensions')

# Check metadata consistency
with open('models/vector_database/metadata.json', 'r') as f:
    metadata = json.load(f)
print(f'✅ Metadata: {len(metadata)} items')

# Check embeddings consistency  
embeddings = np.load('models/vector_database/embeddings.npy')
print(f'✅ Embeddings: {embeddings.shape[0]} vectors match metadata')

# Verify contextual text exists
has_contextual = all('contextual_text' in item for item in metadata)
print(f'✅ Enhanced Context: {\"YES\" if has_contextual else \"NO\"}')
"
```

### View Enhanced Contextual Text
```bash
# From strustore-vector-classification directory
# Display enhanced contextual text for first 3 items
python -c "
import json
with open('models/vector_database/metadata.json', 'r') as f:
    items = json.load(f)
for item in items[:3]:
    print(f'ID: {item[\"id\"]}')
    print(f'Enhanced: {item[\"contextual_text\"]}')
    print('---')
"
```

### Inspect Individual Vectors
```bash
# From strustore-vector-classification directory
# View first embedding vector (768 dimensions)
python -c "
import numpy as np
embeddings = np.load('models/vector_database/embeddings.npy')
print('Vector 0 (full 768 dimensions):')
print(embeddings[0])
"
```

### Compare Vector Similarities
```bash
# From strustore-vector-classification directory
# Compare similarity between items using dot product
python -c "
import numpy as np
import json

# Load embeddings and metadata
embeddings = np.load('models/vector_database/embeddings.npy')
with open('models/vector_database/metadata.json', 'r') as f:
    metadata = json.load(f)

print('=== Vector Similarity Comparison ===')
print()

# Show the first 3 items
for i in range(3):
    item = metadata[i]
    print(f'Item {i}: {item[\"name\"]} (ID: {item[\"id\"]})')
    print(f'  Category: {item[\"category\"]}')
    print()

print('Pairwise Similarities (using dot product):')
print('-' * 40)

# Compare similarities between first 3 items
for i in range(3):
    for j in range(i+1, 3):
        similarity = np.dot(embeddings[i], embeddings[j])
        item_i = metadata[i]['name']
        item_j = metadata[j]['name']
        print(f'Item {i} vs Item {j}: {similarity:.4f}')
        print(f'  \"{item_i}\" vs \"{item_j}\"')
        print()
"
```

---

## Common Issues & Directory Management

### Working Directory Requirements
```bash
# CRITICAL: Always run commands from strustore-vector-classification directory
# From strustore home (where you see: (strustore_env) jovitaeliana@Jo strustore %):
cd strustore-vector-classification

# Verify you're in the right place:
pwd
# Should show: /Users/jovitaeliana/Personal/strustore/strustore-vector-classification

# Check required files exist:
ls models/vector_database/
# Should show: database_config.json, embeddings.npy, faiss_index.bin, item_lookup.json, metadata.json

# If you see errors like "models/: No such file or directory", you're in the wrong directory!
```

### Common Directory Confusion
```bash
# ❌ WRONG - You're in parent strustore directory:
(strustore_env) jovitaeliana@Jo strustore % 

# ✅ CORRECT - You should be in the subdirectory:
(strustore_env) jovitaeliana@Jo strustore-vector-classification %

# Fix by running:
cd strustore-vector-classification
```

### Troubleshooting
- **Mutex warnings**: `[mutex.cc : 452] RAW: Lock blocking` - Normal when using sentence transformers, can be ignored
- **File not found**: Make sure you're in `strustore-vector-classification` directory, not the parent `strustore` directory
- **Model configuration error**: The local `gaming-console-semantic-model` may have config issues. Use `intfloat/multilingual-e5-base` instead
- **Import errors**: Ensure you've activated the virtual environment: `source ../strustore_env/bin/activate`
- **Wrong directory**: Always verify with `pwd` - should show `/Users/jovitaeliana/Personal/strustore/strustore-vector-classification`

---

## Position-Weighted Classification

### Overview
The position-weighted classification system enhances GDINO token analysis by applying exponential decay weighting based on token position. This leverages the fact that GDINO tokens are ranked by confidence, with the highest confidence tokens appearing first.

### Key Features
- **Exponential Decay Weighting**: Position 0 = weight 1.0, position 1 = weight 0.9, etc.
- **Top-8 Token Prioritization**: Higher weights for the most confident tokens
- **Hardware-Specific Enhancement**: Additional weighting for gaming hardware terms
- **Semantic Importance Scoring**: Model numbers and brand terms receive boost

### Position Weighting Formula
```
weight = max(min_weight, exp(-decay_rate × position)) × semantic_importance

where:
- decay_rate = 0.1 (10% decay per position)
- min_weight = 0.1 (minimum threshold)
- semantic_importance = 1.0-1.5 (hardware terms get 1.5x boost)
```

### Example Token Analysis
For tokens: `["nintendo", "dsi", "nintendo dsi", "console", "black", "japan", "tested", "charger"]`

| Position | Token | Position Weight | Semantic Weight | Final Weight | Reason |
|----------|-------|----------------|-----------------|--------------|--------|
| 0 | nintendo | 1.10 | 1.5 | 1.65 | Hardware term + top position |
| 1 | dsi | 0.99 | 1.5 | 1.49 | Hardware term + high position |
| 2 | nintendo dsi | 0.89 | 1.2 | 1.07 | Compound hardware term |
| 3 | console | 0.80 | 1.5 | 1.20 | Hardware descriptor |
| 4 | black | 0.72 | 1.0 | 0.72 | Color term |
| 5 | japan | 0.65 | 1.0 | 0.65 | Geographic term |
| 6 | tested | 0.58 | 1.0 | 0.58 | Condition term |
| 7 | charger | 0.52 | 1.5 | 0.78 | Hardware accessory |

### Configuration Options
```python
from position_weighted_embeddings import PositionWeightedTokenClassifier

# Default configuration
classifier = PositionWeightedTokenClassifier(
    decay_rate=0.1,        # Exponential decay rate
    top_k_boost=8,         # Number of top tokens to prioritize
    min_weight=0.1,        # Minimum weight threshold
    hardware_boost=1.5     # Multiplier for hardware terms
)

# Custom configuration for more aggressive weighting
classifier = PositionWeightedTokenClassifier(
    decay_rate=0.15,       # Faster decay (more emphasis on top tokens)
    top_k_boost=5,         # Focus on top 5 tokens only
    min_weight=0.05,       # Lower minimum weight
    hardware_boost=2.0     # Higher hardware term boost
)
```

### Hardware Terms Recognition
The system recognizes these categories of hardware-specific terms:

**Console Brands**: nintendo, sony, microsoft, sega, playstation, xbox
**Console Types**: ds, dsi, 3ds, switch, wii, ps2, ps3, ps4, gamecube
**Model Numbers**: dol-001, ntr-001, usg-001, twl-001, scph-1001
**Hardware Descriptors**: console, handheld, controller, system

---

## Testing Commands

### Run Real Classification Tests
```bash
# From strustore-vector-classification directory
# Test the complete classification pipeline using GroundingDINO outputs
python src/test_real_classification.py

# Or from strustore home directory:
cd strustore-vector-classification && python src/test_real_classification.py
```

### Run Advanced Position-Weighted Classification Tests
```bash
# From strustore home directory (not strustore-vector-classification)
# Test the new position-weighted classification system
python test_advanced_classifier.py

# This will:
# - Analyze GDINO token ranking patterns
# - Compare position-weighted vs baseline classification
# - Generate comprehensive performance reports
# - Create visualization plots in advanced_test_results/
```

### Test Position-Weighted Token Analysis
```bash
# From strustore home directory
# Demonstrate position weighting on example tokens
python -c "
from position_weighted_embeddings import demonstrate_position_weighting
demonstrate_position_weighting()
"

# Or analyze a specific GDINO file
python -c "
from position_weighted_embeddings import analyze_gdino_file_with_position_weighting
results = analyze_gdino_file_with_position_weighting('gdinoOutput/final/1/r1172860507.json')
print('Position-weighted analysis:', results)
"
```

### Test Vector Database Loading
```bash
# From strustore-vector-classification directory
# Test if the database loads correctly
python -c "
from src.create_vector_database import VectorDatabaseLoader
loader = VectorDatabaseLoader('models/vector_database', 'intfloat/multilingual-e5-base')
loader.load_database()
print('✅ Database loads successfully')
"
```

### Run Semantic Search Test
```bash
# From strustore-vector-classification directory
# Note: The local gaming-console-semantic-model may have configuration issues
# Use the base model instead for testing:

python -c "
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

print('Loading base multilingual model...')
model = SentenceTransformer('intfloat/multilingual-e5-base')
index = faiss.read_index('models/vector_database/faiss_index.bin')
with open('models/vector_database/metadata.json', 'r') as f:
    metadata = json.load(f)

# Test search
query = 'nintendo console'
query_embedding = model.encode([query]).astype('float32')
scores, indices = index.search(query_embedding, 5)

print(f'\\nSearch results for: \"{query}\"')
for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
    item = metadata[idx]
    print(f'{i+1}. {item[\"name\"]} (similarity: {score:.4f})')
"

# Alternative: Simple database test without sentence transformers (ALWAYS WORKS)
python -c "
import faiss
import json
import numpy as np

print('=== Database Files Test ===')
index = faiss.read_index('models/vector_database/faiss_index.bin')
print(f'✅ FAISS Index loaded: {index.ntotal} vectors')

with open('models/vector_database/metadata.json', 'r') as f:
    metadata = json.load(f)
print(f'✅ Metadata loaded: {len(metadata)} items')

embeddings = np.load('models/vector_database/embeddings.npy')
print(f'✅ Embeddings loaded: {embeddings.shape}')

print(f'\\nFirst item: {metadata[0][\"name\"]}')
print(f'Enhanced text: {metadata[0][\"contextual_text\"][:80]}...')
"
```

---

## Codebase Overview

### Project Structure
```
strustore-vector-classification/
├── src/
│   ├── create_vector_database.py    # Creates and manages vector database
│   ├── test_real_classification.py  # End-to-end classification testing
│   └── data/
│       └── training_data/
│           └── triplets.csv         # Training data for semantic model
├── models/
│   ├── gaming-console-semantic-model/  # Trained sentence transformer
│   └── vector_database/               # FAISS index and metadata
│       ├── faiss_index.bin           # FAISS vector index
│       ├── metadata.json             # Item metadata with contextual text
│       ├── embeddings.npy            # Raw embedding vectors
│       ├── item_lookup.json          # Fast item ID lookup
│       └── database_config.json      # Database configuration
└── config/                          # Configuration files
```

### Key Components

#### 1. **Vector Database Creation** (`src/create_vector_database.py`)
- **Purpose**: Creates FAISS vector database from gaming item data
- **Features**:
  - Loads gaming console/accessory data
  - Enhances item descriptions with contextual information
  - Generates 768-dimensional embeddings using multilingual-e5-base
  - Creates FAISS index for fast similarity search
  - Exports metadata with enhanced contextual text

#### 2. **Classification Testing** (`src/test_real_classification.py`)
- **Purpose**: Tests classification pipeline using real GroundingDINO detection results
- **Features**:
  - Loads trained semantic model and vector database
  - Processes GroundingDINO output files from `gdinoOutput/` directory
  - Maps detected labels to canonical taxonomy items
  - Calculates performance metrics and similarity scores
  - Supports multi-language terms (English/Japanese)

#### 3. **Training Data** (`src/data/training_data/triplets.csv`)
- **Format**: Anchor, Positive, Negative triplets for semantic learning
- **Purpose**: Trains the model to understand gaming terminology relationships
- **Content**: Gaming console names, accessories, conditions, colors

---

## Vector Embedding Explanation

### What are Vector Embeddings?
Vector embeddings are numerical representations of text that capture semantic meaning in a high-dimensional space. Each item in the database is converted into a 768-dimensional vector where similar items have similar vectors.

### Our Implementation

#### **Model Used**: `intfloat/multilingual-e5-base`
- **Dimensions**: 768
- **Language Support**: Multilingual (English, Japanese, etc.)
- **Type**: Sentence transformer optimized for semantic similarity

#### **Enhanced Contextual Text Format**:
```
{Item Name} | id: {Item ID} | category: {Category} | {Console Brand} | {Gaming Context Keywords}
```

**Example**:
```
GC Con Loose Stick | id: 00GeemSSwZjPuDwsNaIW | category: Controllers & Attachments | GameCube | gaming controller device | gamepad peripheral
```

#### **Vector Properties**:
- **Range**: Typically -0.1 to +0.1 (normalized)
- **Similarity Calculation**: Cosine similarity via dot product
- **Storage**: FAISS IndexFlatIP for fast inner product search
- **Indexing**: Direct mapping to metadata array positions

### Similarity Examples from Our Database:

| Comparison | Similarity Score | Explanation |
|------------|------------------|-------------|
| "GC Con Loose Stick" vs "GC Mic" | 0.9297 | Both GameCube accessories, same category |
| "Xbox 360 Power Cord" vs "GC Mic" | 0.8485 | Different consoles, different categories |
| "GC Con Loose Stick" vs "Xbox 360 Power Cord" | 0.8331 | Different brands and categories |

---

## File Structure Analysis

### Database Files Breakdown

#### **`models/vector_database/faiss_index.bin`** (712KB)
- **Type**: Binary FAISS index file
- **Content**: 232 vectors × 768 dimensions
- **Purpose**: Fast similarity search using inner product
- **Access**: `faiss.read_index()` function

#### **`models/vector_database/metadata.json`** (86KB)
- **Type**: JSON metadata file
- **Content**: Array of 232 item objects
- **Fields per item**:
  ```json
  {
    "id": "unique_item_id",
    "name": "item_name",
    "category": "item_category", 
    "contextual_text": "enhanced_description_for_embedding"
  }
  ```

#### **`models/vector_database/embeddings.npy`** (712KB)
- **Type**: NumPy array file
- **Shape**: (232, 768)
- **Data Type**: float32
- **Content**: Raw embedding vectors corresponding to metadata items

#### **`models/vector_database/item_lookup.json`** (91KB)
- **Type**: JSON lookup dictionary
- **Structure**: `{"item_id": {item_metadata}}`
- **Purpose**: Fast item retrieval by ID

#### **`models/vector_database/database_config.json`** (688 bytes)
- **Type**: Configuration file
- **Content**: Database creation settings and model information

### Test Data Source

The `test_real_classification.py` script processes detection results from:
```
gdinoOutput/*.json
```

Each JSON file contains:
```json
{
  "image_path": "path/to/image",
  "detections": [
    {
      "box": [x1, y1, x2, y2],
      "confidence": 0.95,
      "label": "detected_term"
    }
  ]
}
```

The script maps these detected labels to the most similar items in the vector database, enabling automatic classification of gaming items from object detection results.

---

## Usage Workflow

1. **Create Database**: Run vector database creation to build FAISS index
2. **Generate Detections**: Use GroundingDINO to detect objects in images
3. **Run Classification**: Test pipeline with `test_real_classification.py`
4. **Analyze Results**: Review similarity scores and classification accuracy
5. **Debug**: Use the commands above to inspect database health and vector similarities

### Expected Test Results
When you run `python src/test_real_classification.py`, you'll see:
- Progress bars for batch processing
- INFO logs showing test progress
- Performance metrics including:
  - Total tests processed (e.g., 15 detections)
  - Overall accuracy percentage
  - Category-wise performance breakdown
  - Similarity threshold analysis
- Visualization plots saved to `test_results/` directory
- Detailed report saved to `classification_test_report.json`

Example output:
```
OVERALL PERFORMANCE:
  Total Tests: 15
  Correct Predictions: 1
  Overall Accuracy: 6.67%
  Average Detection Confidence: 0.865
  Average Top-1 Similarity: 0.803

CATEGORY-WISE PERFORMANCE:
  Consoles: 0/4 (0.00%)
  Controllers: 0/2 (0.00%)
  Model_Codes: 1/1 (100.00%)
  ...
```

This system enables semantic search and classification of gaming items using state-of-the-art NLP models and efficient vector similarity search.

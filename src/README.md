# Source Code (`src/`)

*Last updated: 24 June 2025*
*Curtis Murray*

---

Reusable model implementations and utilities for VetBERT antimicrobial prescription classification.

## Directory Structure

```
src/
├── models/
│   └── vetbert.py          # VetBERT wrapper class
└── evaluation/             # Evaluation utilities and frameworks
```

## Models (`models/`)

### `vetbert.py` - VetBERT Wrapper Class

**Purpose**: Wrapper for VetBERT base model from Hugging Face
**Model**: `havocy28/VetBERTDx` 

**Key Features**:
- Soft predictions for ~50 veterinary syndromes
- Batch processing capabilities
- Standardised text preprocessing
- Configurable prediction thresholds

**Usage**:
```python
from src.models.vetbert import VetBERT

# Initialize model
model = VetBERT()

# Generate predictions for single text
prediction = model.predict_single("dog with urinary symptoms")

# Batch processing for DataFrame
predictions = model.classify_texts(df, input_column="ExaminationText")
```

**Used By**:
- `scripts/prediction/01_predict_vetbert_on_sophie_dataset.py`
- `scripts/prediction/01_predict_vetbert_original.py`

## Evaluation (`evaluation/`)

**Purpose**: Evaluation utilities and performance measurement frameworks

## Code Organisation Principles

### Reusability
- **`src/`**: Code designed for reuse across multiple scripts
- **`scripts/`**: Task-specific implementations and workflows
- **Import Pattern**: Scripts import from `src/` for core functionality
*Largely followed, there may be exceptions*

## Usage Patterns
```
VetBERT Base Model (havocy28/VetBERTDx)
    ↓
src/models/vetbert.py (wrapper for base model)
    ↓  
scripts/training/*.py (fine-tuned classification heads)
    ↓
scripts/prediction/*.py (inference with fine-tuned models)
```

### Standard VetBERT Predictions
```python
# Import wrapper
from src.models.vetbert import VetBERT

# Initialize and use
model = VetBERT()
results = model.classify_texts(data_df, input_column="ExaminationText")
```

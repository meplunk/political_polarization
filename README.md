# Political Polarization NLP Pipeline

This pipeline trains machine learning models to predict political ideology from congressional speeches using TF-IDF vectorization and abstract embeddings.

## Overview

The pipeline uses speeches from the 114th Congress along with DIME ideology scores to build predictive models. Two separate models are trained:
1. **TF-IDF Model**: Uses TF-IDF vectorization with cross-validation over both vectorizer parameters and Ridge regression alpha
2. **Embeddings Model**: Uses pre-computed abstract embeddings with cross-validation over Ridge regression alpha only

## Data Sources

This project uses the following datasets (see `data/01_raw/` for details):

1. **DIME Scores** (`dime_recipients_1979_2024.csv`)
   - Citation: Bonica, Adam. 2024. Database on Ideology, Money in Politics, and Elections: Public version 4.0. Stanford University Libraries. https://data.stanford.edu/dime

2. **Congressional Speeches** (`speeches_114.txt`, `114_SpeakerMap.txt`)
   - Citation: Gentzkow, Matthew, Jesse M. Shapiro, and Matt Taddy. Congressional Record for the 43rd-114th Congresses: Parsed Speeches and Phrase Counts. Stanford Libraries, 2018. https://data.stanford.edu/congress_text

3. **Political TV Advertisements** (`2016HouseVideo/`)

4. **WMP TV Ad Metadata** (`wmp-house-2016-v1.0.dta`)

5. **Election Results** (`medsl_data_clean.dta`)

## Requirements

- Python 3.10.5+
- Required packages listed in `requirements.txt`

## Setup

1. **Clone the repository**
```bash
git clone https://github.com/meplunk/political_polarization.git
cd political_polarization
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Add raw data**
   - Place all raw data files in `data/01_raw/`
   - The pipeline will create other data folders automatically

## Project Structure
```
project_root/
├── data/
│   ├── 01_raw/              # Original datasets (not in Git)
│   ├── 02_cleaned/          # Cleaned speeches and merged data
│   ├── 03_features/         # TF-IDF vectors and embeddings
│   └── 04_predictions/      # Model predictions
├── models/
│   ├── tfidf_pipeline.pkl   # Trained TF-IDF model
│   └── embeddings_pipeline.pkl  # Trained embeddings model
└── code/
    ├── config.py            # Paths and hyperparameters
    ├── stage_01_clean.py    # Text cleaning and preprocessing
    ├── stage_02_vectorize.py  # Feature engineering
    ├── stage_03_train.py    # Model training with CV
    ├── stage_04_predict.py  # Generate predictions
    ├── run_all.py           # Run complete pipeline
    └── utils.py             # Helper functions
```

## Usage

### Run Complete Pipeline

To run all stages from start to finish:
```bash
python code/run_all.py
```

This will:
1. Clean and preprocess congressional speeches
2. Create TF-IDF features and load embeddings
3. Train both models with cross-validation
4. Generate predictions on test data

### Run Individual Stages

You can also run stages independently:
```bash
# Stage 1: Clean and preprocess text
python code/stage_01_clean.py

# Stage 2: Create features (TF-IDF and embeddings)
python code/stage_02_vectorize.py

# Stage 3: Train models
python code/stage_03_train.py

# Stage 4: Generate predictions
python code/stage_04_predict.py
```

**Benefits of running stages separately:**
- Debug issues at specific steps
- Skip expensive stages if data hasn't changed
- Experiment with different parameters in `config.py`

## Configuration

Modify hyperparameters and paths in `code/config.py`:
```python
# TF-IDF hyperparameter grid
TFIDF_PARAMS = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'ridge__alpha': [0.1, 1.0, 10.0]
}

# Embeddings only searches over alpha
EMBEDDINGS_PARAMS = {
    'ridge__alpha': [0.1, 1.0, 10.0]
}
```

## Output

After running the pipeline, you'll find:

- **Trained Models**: `models/tfidf_pipeline.pkl` and `models/embeddings_pipeline.pkl`
- **Predictions**: `data/04_predictions/tfidf_predictions.csv` and `embeddings_predictions.csv`
- **Cross-Validation Results**: Logged during training with best parameters and scores

## Model Details

### TF-IDF Model
- **Vectorization**: TF-IDF with configurable n-grams and max features
- **Regression**: Ridge regression with L2 regularization
- **Cross-Validation**: Grid search over vectorizer and regression parameters

### Embeddings Model
- **Features**: Pre-computed abstract embeddings
- **Regression**: Ridge regression with L2 regularization
- **Cross-Validation**: Grid search over alpha parameter only

## Troubleshooting

**Missing data folders?**
- The pipeline creates folders automatically. If you get errors, ensure `data/01_raw/` exists with source files.

**Import errors?**
- Run `pip install -r requirements.txt` to install all dependencies

**Memory issues?**
- Reduce `max_features` in `config.py` for TF-IDF model
- Process data in smaller batches

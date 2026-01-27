# config.py
import pandas as pd
from pathlib import Path
import pickle


# ============================================
# BASE PATHS
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CODE_DIR = PROJECT_ROOT / "code"

# ============================================
# RAW DATA PATHS
# ============================================
RAW_DIR = DATA_DIR / "01_raw"

# DIME ideology scores
DIME_DATA = RAW_DIR / "dime_recipients_1979_2024.csv"

# Congressional speeches
SPEECHES_FILE = RAW_DIR / "speeches_114.txt"
SPEAKER_MAP = RAW_DIR / "114_SpeakerMap.txt"

# TV ad data
VIDEO_DIR = RAW_DIR / "2016HouseVideo"
WMP_METADATA = RAW_DIR / "wmp-house-2016-v1.0.dta"

# Election results
ELECTION_DATA = RAW_DIR / "medsl_data_clean.dta"

# ============================================
# PROCESSED DATA PATHS
# ============================================
CLEANED_DIR = DATA_DIR / "02_cleaned"
FEATURES_DIR = DATA_DIR / "03_features"
PREDICTIONS_DIR = DATA_DIR / "04_predictions"

# Cleaned/merged data
CLEANED_SPEECHES = CLEANED_DIR / "cleaned_speeches.csv"
MERGED_DATA = CLEANED_DIR / "merged_speeches_ideology.csv"

# Features
TFIDF_FEATURES = FEATURES_DIR / "tfidf_features.npz"
EMBEDDINGS_FEATURES = FEATURES_DIR / "embeddings.npy"

# Predictions
TFIDF_PREDICTIONS = PREDICTIONS_DIR / "tfidf_predictions.csv"
EMBEDDINGS_PREDICTIONS = PREDICTIONS_DIR / "embeddings_predictions.csv"

# ============================================
# MODEL PATHS
# ============================================
TFIDF_MODEL = MODELS_DIR / "tfidf_pipeline.pkl"
EMBEDDINGS_MODEL = MODELS_DIR / "embeddings_pipeline.pkl"

# CV results
TFIDF_CV_RESULTS = MODELS_DIR / "tfidf_cv_results.csv"
EMBEDDINGS_CV_RESULTS = MODELS_DIR / "embeddings_cv_results.csv"

# ============================================
# DATA COLUMN NAMES
# ============================================
TEXT_COLUMN = "speech"
TARGET_COLUMN = "dime"
SPEAKER_ID_COLUMN = "unique_id"
AD_TEXT_COLUMN = "ad_text"
AD_ID_COLUMN = "vidfile"

# ============================================
# HYPERPARAMETERS
# ============================================

# TF-IDF Model - CV over vectorizer AND regression params
TFIDF_PARAMS = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__min_df': [2, 5],
    'ridge__alpha': [0.1, 1.0, 10.0, 100.0]
}

# Embeddings Model - CV over regression params only
EMBEDDINGS_PARAMS = {
    'ridge__alpha': [0.1, 1.0, 10.0, 100.0]
}

# Cross-validation settings
CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================
# TEXT CLEANING PARAMETERS
# ============================================
# Minimum speech length (in characters) to include
MIN_SPEECH_LENGTH = 50

# Remove stopwords?
REMOVE_STOPWORDS = True

# Lowercase text?
LOWERCASE = True


def save_pickle(obj, filepath):
    """Save object to pickle file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved to {filepath}")


def load_pickle(filepath):
    """Load object from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def create_sparse_dataframe(X, index, feature_names):
    """Create sparse DataFrame from scipy sparse matrix"""
    return pd.DataFrame.sparse.from_spmatrix(
        X,
        index=index,
        columns=feature_names
    )

{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww19540\viewh13820\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 """\
===============================================================================\
FILE: stage_02_tokenize.py\
AUTHOR: Mary Edith Plunkett\
PROJECT: Political Polarization Project\
DATE: January 27, 2026\
===============================================================================\
PURPOSE:\
    Tokenize and preprocess text data using spaCy NLP pipeline.\
    This is Stage 2 of the NLP pipeline.\
\
DESCRIPTION:\
    1. Load cleaned speeches and unique ads from Stage 1\
    2. Apply spaCy linguistic preprocessing:\
       - Tokenization\
       - Lemmatization\
       - Stopword removal (if enabled in config)\
       - Punctuation removal\
       - Lowercasing (if enabled in config)\
    3. Save tokenized speeches and ads for vectorization\
    \
    Note: Only unique ads (~1-2K) are tokenized, not all airings (~300-400K).\
    This is efficient since we only need the vocabulary from unique ad text.\
\
INPUT FILES:\
    - data/02_cleaned/cleaned_speeches.csv\
    - data/02_cleaned/cleaned_ads_unique.csv\
\
OUTPUT FILES:\
    - data/02_cleaned/tokenized_speeches.csv\
    - data/02_cleaned/tokenized_ads_unique.csv\
\
DEPENDENCIES:\
    - spacy (with en_core_web_sm model)\
    - pandas\
    - tqdm\
\
USAGE:\
    python code/stage_02_tokenize.py\
===============================================================================\
"""\
\
import pandas as pd\
import spacy\
from tqdm import tqdm\
from pathlib import Path\
from config import (\
    CLEANED_SPEECHES, CLEANED_ADS_UNIQUE,\
    TOKENIZED_SPEECHES, TOKENIZED_ADS,\
    CLEANED_DIR,\
    TEXT_COLUMN, TARGET_COLUMN,\
    AD_TEXT_COLUMN, AD_ID_COLUMN,\
    REMOVE_STOPWORDS, LOWERCASE\
)\
\
# Load spaCy's English language model\
# Disable NER and parser for faster processing (we only need tokenization)\
print("Loading spaCy model...")\
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])\
\
\
def tokenize_text(text_series, batch_size=500):\
    """\
    Preprocess text using spaCy linguistic pipeline.\
    \
    Performs:\
    - Tokenization: split text into words\
    - Lowercasing: normalize text (if enabled in config)\
    - Stopword removal: remove common words (if enabled in config)\
    - Punctuation removal: keep only alphabetic tokens\
    - Lemmatization: reduce words to base form\
    \
    Example:\
        "The politicians are running for office"\
        -> ["politician", "run", "office"]\
    \
    Args:\
        text_series: pandas Series or list of text documents\
        batch_size: Number of texts to process simultaneously\
        \
    Returns:\
        List of lists containing cleaned tokens\
    """\
    cleaned_texts = []\
    \
    # Process texts in batches using spaCy's pipeline\
    for doc in tqdm(\
        nlp.pipe(text_series, batch_size=batch_size),\
        total=len(text_series),\
        desc="Tokenizing text"\
    ):\
        # Extract tokens that meet criteria\
        tokens = []\
        for token in doc:\
            # Skip non-alphabetic tokens (numbers, punctuation)\
            if not token.is_alpha:\
                continue\
            \
            # Skip stopwords if enabled in config\
            if REMOVE_STOPWORDS and token.is_stop:\
                continue\
            \
            # Get lemmatized form (base form of word)\
            word = token.lemma_\
            \
            # Lowercase if enabled in config\
            if LOWERCASE:\
                word = word.lower()\
            \
            tokens.append(word)\
        \
        cleaned_texts.append(tokens)\
    \
    return cleaned_texts\
\
\
def tokenize_speeches():\
    """\
    Load and tokenize Congressional speeches.\
    \
    Loads speeches from Stage 1, applies tokenization, and saves\
    the result with tokens joined back into strings for storage.\
    \
    Returns:\
        DataFrame with tokenized speeches\
    """\
    print("Loading cleaned speeches...")\
    \
    # Verify input file exists\
    if not CLEANED_SPEECHES.exists():\
        raise FileNotFoundError(\
            f"Cleaned speeches not found at \{CLEANED_SPEECHES\}\\n"\
            f"Please run stage_01_clean.py first."\
        )\
    \
    # Load the cleaned speeches from Stage 1\
    df = pd.read_csv(CLEANED_SPEECHES)\
    print(f"  Loaded \{len(df):,\} speeches")\
    \
    # Apply tokenization to all speeches\
    print("Applying tokenization...")\
    tokenized = tokenize_text(df[TEXT_COLUMN])\
    \
    # Join tokens back into strings for storage\
    # (easier to work with in CSV format than lists)\
    df['tokenized_speech'] = [' '.join(tokens) for tokens in tokenized]\
    \
    # Also keep token count for reference\
    df['token_count'] = [len(tokens) for tokens in tokenized]\
    \
    print(f"  Tokenization complete: \{len(df):,\} speeches processed")\
    print(f"  Average tokens per speech: \{df['token_count'].mean():.1f\}")\
    \
    return df\
\
\
def tokenize_ads():\
    """\
    Load and tokenize unique ad transcripts.\
    \
    Only tokenizes unique ads (~1-2K), not all airings (~300-400K).\
    This is efficient for vocabulary building.\
    \
    Returns:\
        DataFrame with tokenized ads (or None if no ads file)\
    """\
    # Check if unique ads file exists\
    if not CLEANED_ADS_UNIQUE.exists():\
        print(f"  No unique ads file found at \{CLEANED_ADS_UNIQUE\}, skipping...")\
        return None\
    \
    print("Loading unique ads...")\
    \
    # Load the unique ads from Stage 1\
    df = pd.read_csv(CLEANED_ADS_UNIQUE)\
    print(f"  Loaded \{len(df):,\} unique ads")\
    \
    # Apply tokenization to all ads\
    print("Applying tokenization...")\
    tokenized = tokenize_text(df[AD_TEXT_COLUMN])\
    \
    # Join tokens back into strings for storage\
    df['tokenized_ad'] = [' '.join(tokens) for tokens in tokenized]\
    \
    # Also keep token count for reference\
    df['token_count'] = [len(tokens) for tokens in tokenized]\
    \
    print(f"  Tokenization complete: \{len(df):,\} unique ads processed")\
    print(f"  Average tokens per ad: \{df['token_count'].mean():.1f\}")\
    \
    return df\
\
\
def main():\
    """\
    Execute the complete tokenization pipeline.\
    """\
    print("\\n" + "="*80)\
    print("STAGE 2: TEXT TOKENIZATION")\
    print("="*80 + "\\n")\
    \
    print(f"Settings:")\
    print(f"  Remove stopwords: \{REMOVE_STOPWORDS\}")\
    print(f"  Lowercase: \{LOWERCASE\}")\
    print()\
    \
    # Tokenize speeches\
    print("Step 1: Tokenizing speeches")\
    print("-" * 40)\
    df_speeches = tokenize_speeches()\
    \
    # Save tokenized speeches\
    print(f"\\nSaving tokenized speeches...")\
    df_speeches.to_csv(TOKENIZED_SPEECHES, index=False)\
    print(f"  Saved to: \{TOKENIZED_SPEECHES\}")\
    \
    # Tokenize ads\
    print("\\nStep 2: Tokenizing ads")\
    print("-" * 40)\
    df_ads = tokenize_ads()\
    \
    # Save tokenized ads (if they exist)\
    if df_ads is not None:\
        print(f"\\nSaving tokenized ads...")\
        df_ads.to_csv(TOKENIZED_ADS, index=False)\
        print(f"  Saved to: \{TOKENIZED_ADS\}")\
    \
    # Summary statistics\
    print("\\n" + "="*80)\
    print("SUMMARY")\
    print("="*80)\
    \
    print(f"\\nSPEECHES:")\
    print(f"  Total speeches: \{len(df_speeches):,\}")\
    print(f"  Average tokens per speech: \{df_speeches['token_count'].mean():.1f\}")\
    print(f"  Median tokens per speech: \{df_speeches['token_count'].median():.0f\}")\
    print(f"  Min tokens: \{df_speeches['token_count'].min()\}")\
    print(f"  Max tokens: \{df_speeches['token_count'].max()\}")\
    \
    if df_ads is not None:\
        print(f"\\nADS:")\
        print(f"  Total unique ads: \{len(df_ads):,\}")\
        print(f"  Average tokens per ad: \{df_ads['token_count'].mean():.1f\}")\
        print(f"  Median tokens per ad: \{df_ads['token_count'].median():.0f\}")\
        print(f"  Min tokens: \{df_ads['token_count'].min()\}")\
        print(f"  Max tokens: \{df_ads['token_count'].max()\}")\
    \
    print(f"\\nOutput files:")\
    print(f"  \{TOKENIZED_SPEECHES\}")\
    if df_ads is not None:\
        print(f"  \{TOKENIZED_ADS\}")\
    \
    print("\\n" + "="*80)\
    print("STAGE 2 COMPLETE")\
    print("="*80 + "\\n")\
\
\
if __name__ == "__main__":\
    main()}
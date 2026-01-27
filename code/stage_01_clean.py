"""
===============================================================================
FILE: stage_01_clean.py
AUTHOR: Mary Edith Plunkett
PROJECT: Political Polarization Project (PPP)
CREATED: October 23, 2025
MODIFIED: January 27, 2026
===============================================================================
PURPOSE:
    Clean and merge Congressional speeches with DIME ideology scores.
    This is Stage 1 of the NLP pipeline.

DESCRIPTION:
    1. Filter DIME recipients to federal candidates (2013-2018)
    2. Merge Congressional speeches with speaker metadata
    3. Link speeches to DIME ideology scores
    4. Create standardized unique_id for merging (LASTNAME_STATE_DISTRICT)

INPUT FILES:
    - data/01_raw/dime_recipients_1979_2024.csv
    - data/01_raw/speeches_114.txt
    - data/01_raw/114_SpeakerMap.txt

OUTPUT FILES:
    - data/02_cleaned/cleaned_speeches.csv

USAGE:
    python code/stage_01_clean.py

===============================================================================
"""
import os
import re
import pandas as pd
from pathlib import Path
from config import (
    DIME_DATA, SPEECHES_FILE, SPEAKER_MAP, VIDEO_DIR, WMP_METADATA,
    CLEANED_SPEECHES, CLEANED_ADS_AIRINGS, CLEANED_ADS_UNIQUE, CLEANED_DIR,
    TEXT_COLUMN, TARGET_COLUMN, SPEAKER_ID_COLUMN,
    AD_TEXT_COLUMN, AD_ID_COLUMN,
    MIN_SPEECH_LENGTH, MIN_AD_LENGTH
)


def filter_dime_recipients(dime_path):
    """
    Filter DIME dataset to federal candidates from 2013-2018.
    
    Args:
        dime_path: Path to raw DIME CSV file
        
    Returns:
        DataFrame with filtered DIME data and unique_id column
    """
    print("Loading DIME recipients data...")
    df = pd.read_csv(dime_path)
    print(f"  Loaded {len(df):,} rows")
    
    # Filter to study period (2013-2018)
    df = df[(df['cycle'] >= 2013) & (df['cycle'] <= 2018)]
    print(f"  After filtering to 2013-2018: {len(df):,} rows")
    
    # Keep only federal House and Senate candidates
    df = df[(df['seat'] == "federal:house") | (df['seat'] == "federal:senate")]
    print(f"  After filtering to federal candidates: {len(df):,} rows")
    
    # Create unique identifier: LASTNAME_STATE_DISTRICT
    # Example: "PELOSI_CA_12"
    df['unique_id'] = (
        df['lname'].str.split().str[0].str.upper() + '_' +
        df['distcyc'].str.split('_', n=1).str[1]
    )
    
    return df


def merge_speakers_and_speeches(speaker_path, speech_path):
    """
    Merge Congressional Record speaker metadata with speech texts.
    
    Args:
        speaker_path: Path to speaker map file
        speech_path: Path to speeches file
        
    Returns:
        DataFrame with merged speaker and speech data
    """
    print("Loading Congressional Record data...")
    
    # Load speaker metadata and speeches (pipe-delimited)
    df_speakers = pd.read_csv(speaker_path, delimiter="|", encoding="latin1")
    df_speeches = pd.read_csv(speech_path, delimiter="|", encoding="latin1",
                              on_bad_lines="warn")
    
    print(f"  Loaded {len(df_speakers):,} speakers")
    print(f"  Loaded {len(df_speeches):,} speeches")
    
    # Merge on speech_id
    df_merged = pd.merge(df_speakers, df_speeches, on="speech_id", how="right")
    
    # Keep essential columns
    keep_cols = ["speakerid", "speech_id", "lastname", "firstname",
                 "speech", "state", "district"]
    df_merged = df_merged[keep_cols]
    
    # Remove rows with missing speaker names
    df_merged = df_merged[df_merged['lastname'].notna() & 
                          (df_merged['lastname'] != '')]
    
    # Create unique_id to match DIME format
    # For House: LASTNAME_STATE_DISTRICT (e.g., "PELOSI_CA_12")
    # For Senate: LASTNAME_STATE_S (e.g., "WARREN_MA_S")
    df_merged['unique_id'] = (
        df_merged['lastname'].str.split().str[0].str.upper() + '_' +
        df_merged['state'].astype(str).str.upper() + '_' +
        df_merged['district'].apply(lambda x: str(int(x)) if pd.notnull(x) else 'S')
    )
    
    # Apply manual corrections for known naming inconsistencies
    df_merged = apply_name_corrections(df_merged)
    
    print(f"  Merged dataset: {len(df_merged):,} speeches")
    
    return df_merged


def apply_name_corrections(df):
    """
    Apply manual corrections for known mismatches between 
    Congressional Record and DIME naming conventions.
    
    Args:
        df: DataFrame with unique_id column
        
    Returns:
        DataFrame with corrected unique_ids
    """
    corrections = {
        # At-large districts (0 -> 1)
        "NORTON_DC_0": "NORTON_DC_1",
        "PIERLUISI_PR_0": "PIERLUISI_PR_1",
        "CARNEY_DE_0": "CARNEY_DE_1",
        "WELCH_VT_0": "WELCH_VT_1",
        "PLASKETT_VI_0": "PLASKETT_VI_1",
        "CRAMER_ND_0": "CRAMER_ND_1",
        "NOEM_SD_0": "NOEM_SD_1",
        "LUMMIS_WY_0": "LUMMIS_WY_1",
        "ZINKE_MT_0": "ZINKE_MT_1",
        "YOUNG_AK_0": "YOUNG_AK_1",
        "SABLAN_MP_0": "SABLAN_MP_1",
        "BORDALLO_GU_0": "BORDALLO_GU_1",
        
        # Hyphenated last names
        "VAN_MD_8": "VANHOLLEN_MD_8",
        "ROS-LEHTINEN_FL_27": "ROS_FL_27",
        "DIAZ-BALART_FL_25": "DIAZ_FL_25",
        "ROYBAL-ALLARD_CA_40": "ROYBAL_CA_40",
        
        # Name changes
        "JACKSON_TX_18": "LEE_TX_18"  # Sheila Jackson Lee
    }
    
    df['unique_id'] = df['unique_id'].replace(corrections)
    return df


def merge_with_ideology_scores(df_speeches, df_dime):
    """
    Merge speeches with DIME ideology scores.
    
    Args:
        df_speeches: DataFrame with speeches and unique_id
        df_dime: DataFrame with DIME scores and unique_id
        
    Returns:
        DataFrame with speeches and ideology scores
    """
    print("Merging speeches with ideology scores...")
    
    # Merge on unique_id
    merged = pd.merge(df_speeches, df_dime, on='unique_id', how='left')
    
    # Rename columns to match config
    column_mapping = {
        'speakerid': SPEAKER_ID_COLUMN,
        'speech': TEXT_COLUMN,
        'dwdime': TARGET_COLUMN
    }
    merged = merged.rename(columns=column_mapping)
    
    # Keep essential columns
    keep_cols = [
        SPEAKER_ID_COLUMN,
        'speech_id',
        TEXT_COLUMN,
        'unique_id',
        'lastname',
        'firstname',
        'state',
        'district',
        TARGET_COLUMN,
        'gen.vote.pct',
        'gwinner'
    ]
    
    # Only keep columns that exist
    keep_cols = [col for col in keep_cols if col in merged.columns]
    merged = merged[keep_cols]
    
    # Drop speeches missing text or ideology scores
    initial_count = len(merged)
    merged = merged.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN])
    print(f"  Dropped {initial_count - len(merged):,} rows with missing text or scores")
    
    # Filter by minimum speech length
    if MIN_SPEECH_LENGTH > 0:
        merged = merged[merged[TEXT_COLUMN].str.len() >= MIN_SPEECH_LENGTH]
        print(f"  Kept speeches with length >= {MIN_SPEECH_LENGTH} characters")
    
    merged = merged.reset_index(drop=True)
    print(f"  Final dataset: {len(merged):,} speeches")
    
    return merged


def load_and_clean_ads(video_dir, metadata_path, df_dime):
    """
    Load ad transcripts and merge with metadata and DIME scores.
    
    Creates two datasets:
    1. All airings - every instance of an ad being aired (~1M rows)
    2. Unique ads - one row per unique ad creative (~2K rows)
    
    Args:
        video_dir: Path to directory with ad transcript .txt files
        metadata_path: Path to WMP metadata file
        df_dime: DataFrame with DIME scores
        
    Returns:
        Tuple of (df_airings, df_unique): DataFrames for airings and unique ads
    """
    print("Loading ad transcripts...")
    
    # Collect all .txt files from video directory
    txt_files = list(Path(video_dir).glob("*.txt"))
    
    if not txt_files:
        print(f"  Warning: No .txt files found in {video_dir}")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"  Found {len(txt_files):,} ad transcript files")
    
    # Read all ad transcripts
    ads_data = []
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            ads_data.append({
                AD_ID_COLUMN: txt_file.stem,  # filename without extension
                AD_TEXT_COLUMN: text
            })
    
    df_ads_text = pd.DataFrame(ads_data)
    print(f"  Loaded {len(df_ads_text):,} unique ad transcripts")
    
    # Load WMP metadata (has one row per airing)
    print("Loading WMP metadata...")
    df_metadata = pd.read_stata(metadata_path)
    print(f"  Loaded {len(df_metadata):,} ad airings")
    
    # Clean metadata
    df_metadata = df_metadata[df_metadata['cand_id'].notna()]
    df_metadata = df_metadata[~df_metadata['cand_id'].astype(str).str.match(r'^\s*$')]
    
    # Create unique_id from metadata
    def make_unique_id(row):
        """Extract candidate name and create unique_id."""
        pattern = r"[_,\- ]"
        last = re.split(pattern, str(row['cand_id']))[0]
        last = last.replace("'", "").upper()
        district = re.sub(r"^0+", "", str(row['district'])) or "0"
        return f"{last}_{row['state']}_{district}"
    
    df_metadata['unique_id'] = df_metadata.apply(make_unique_id, axis=1)
    
    # Apply manual corrections for ad metadata
    ad_corrections = {
        "LINDBECK_AK_1": "LINDBECK_AK_NA",
        "DAVID_IA_3": "YOUNG_IA_3",
        "ANDY_KY_6": "BARR_KY_6",
        "BEUTLER_WA_3": "HERRERA_WA_3",
        "RODGERS_WA_5": "MCMORRIS_WA_5"
    }
    df_metadata['unique_id'] = df_metadata['unique_id'].replace(ad_corrections)
    
    # Merge metadata with DIME scores
    df_metadata = pd.merge(df_metadata, df_dime[['unique_id', 'dwdime']], 
                           on='unique_id', how='left')
    df_metadata = df_metadata.rename(columns={'dwdime': TARGET_COLUMN})
    
    # Create AIRINGS dataset: metadata for all airings
    airings_cols = [AD_ID_COLUMN, 'unique_id', TARGET_COLUMN, 'party', 
                    'race_id', 'biweek']
    airings_cols = [col for col in airings_cols if col in df_metadata.columns]
    df_airings = df_metadata[airings_cols].copy()
    
    # Drop airings missing scores
    initial_airings = len(df_airings)
    df_airings = df_airings.dropna(subset=[TARGET_COLUMN])
    print(f"  Kept {len(df_airings):,} airings with ideology scores "
          f"(dropped {initial_airings - len(df_airings):,})")
    
    # Create UNIQUE ADS dataset: one row per ad with text
    # Merge unique ad text with metadata (using first occurrence of each ad)
    df_unique = df_metadata.drop_duplicates(subset=[AD_ID_COLUMN], keep='first')
    df_unique = pd.merge(df_ads_text, df_unique, on=AD_ID_COLUMN, how='left')
    
    # Keep essential columns for unique ads
    unique_cols = [AD_ID_COLUMN, AD_TEXT_COLUMN, 'unique_id', TARGET_COLUMN, 'party']
    unique_cols = [col for col in unique_cols if col in df_unique.columns]
    df_unique = df_unique[unique_cols]
    
    # Drop unique ads missing text or scores
    initial_unique = len(df_unique)
    df_unique = df_unique.dropna(subset=[AD_TEXT_COLUMN, TARGET_COLUMN])
    print(f"  Kept {len(df_unique):,} unique ads with text and scores "
          f"(dropped {initial_unique - len(df_unique):,})")
    
    # Filter by minimum length
    if MIN_AD_LENGTH > 0:
        df_unique = df_unique[df_unique[AD_TEXT_COLUMN].str.len() >= MIN_AD_LENGTH]
        print(f"  Kept ads with length >= {MIN_AD_LENGTH} characters")
    
    df_airings = df_airings.reset_index(drop=True)
    df_unique = df_unique.reset_index(drop=True)
    
    return df_airings, df_unique


def main():
    """
    Execute the complete Stage 1 cleaning pipeline.
    """
    print("\n" + "="*80)
    print("STAGE 1: DATA CLEANING")
    print("="*80 + "\n")
    
    # Create output directory if it doesn't exist
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Filter DIME recipients
    print("Step 1: Filtering DIME recipients")
    print("-" * 40)
    df_dime = filter_dime_recipients(DIME_DATA)
    
    # Step 2: Merge speakers and speeches
    print("\nStep 2: Merging speakers and speeches")
    print("-" * 40)
    df_speeches = merge_speakers_and_speeches(SPEAKER_MAP, SPEECHES_FILE)
    
    # Step 3: Merge with ideology scores
    print("\nStep 3: Merging speeches with ideology scores")
    print("-" * 40)
    df_speeches_final = merge_with_ideology_scores(df_speeches, df_dime)
    
    # Step 4: Load and clean ads
    print("\nStep 4: Loading and cleaning ad transcripts")
    print("-" * 40)
    df_ads_airings, df_ads_unique = load_and_clean_ads(VIDEO_DIR, WMP_METADATA, df_dime)
    
    # Step 5: Save cleaned data
    print("\nStep 5: Saving cleaned data")
    print("-" * 40)
    df_speeches_final.to_csv(CLEANED_SPEECHES, index=False)
    print(f"  Speeches saved to: {CLEANED_SPEECHES}")
    
    if len(df_ads_airings) > 0:
        df_ads_airings.to_csv(CLEANED_ADS_AIRINGS, index=False)
        print(f"  Ad airings saved to: {CLEANED_ADS_AIRINGS}")
    
    if len(df_ads_unique) > 0:
        df_ads_unique.to_csv(CLEANED_ADS_UNIQUE, index=False)
        print(f"  Unique ads saved to: {CLEANED_ADS_UNIQUE}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nSPEECHES:")
    print(f"  Total speeches: {len(df_speeches_final):,}")
    print(f"  Unique speakers: {df_speeches_final[SPEAKER_ID_COLUMN].nunique():,}")
    print(f"  Mean ideology score: {df_speeches_final[TARGET_COLUMN].mean():.3f}")
    
    if len(df_ads_airings) > 0:
        print(f"\nAD AIRINGS:")
        print(f"  Total airings: {len(df_ads_airings):,}")
        print(f"  Unique ads: {df_ads_airings[AD_ID_COLUMN].nunique():,}")
        print(f"  Mean ideology score: {df_ads_airings[TARGET_COLUMN].mean():.3f}")
    
    if len(df_ads_unique) > 0:
        print(f"\nUNIQUE ADS (for tokenization):")
        print(f"  Total unique ads: {len(df_ads_unique):,}")
        print(f"  Unique candidates: {df_ads_unique['unique_id'].nunique():,}")
        print(f"  Mean ideology score: {df_ads_unique[TARGET_COLUMN].mean():.3f}")
    
    print(f"\nOutput files:")
    print(f"  {CLEANED_SPEECHES}")
    if len(df_ads_airings) > 0:
        print(f"  {CLEANED_ADS_AIRINGS}")
    if len(df_ads_unique) > 0:
        print(f"  {CLEANED_ADS_UNIQUE}")
    
    print("\n" + "="*80)
    print("STAGE 1 COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


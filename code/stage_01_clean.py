"""
===============================================================================
FILE: stage_01_clean.py
AUTHOR: Mary Edith Plunkett
PROJECT: Political Polarization Project (PPP)
DATE: October 23, 2025
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
import pandas as pd
from pathlib import Path
from config import (
    DIME_DATA, SPEECHES_FILE, SPEAKER_MAP,
    CLEANED_SPEECHES, CLEANED_DIR,
    TEXT_COLUMN, TARGET_COLUMN, SPEAKER_ID_COLUMN,
    MIN_SPEECH_LENGTH
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
        'unique_id': SPEAKER_ID_COLUMN,
        'speech': TEXT_COLUMN,
        'dwdime': TARGET_COLUMN
    }
    merged = merged.rename(columns=column_mapping)
    
    # Keep essential columns
    keep_cols = [
        SPEAKER_ID_COLUMN,
        'speech_id',
        TEXT_COLUMN,
        'Speakerid',
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
    print("\nStep 3: Merging with ideology scores")
    print("-" * 40)
    df_final = merge_with_ideology_scores(df_speeches, df_dime)
    
    # Step 4: Save cleaned data
    print("\nStep 4: Saving cleaned data")
    print("-" * 40)
    df_final.to_csv(CLEANED_SPEECHES, index=False)
    print(f"  Saved to: {CLEANED_SPEECHES}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total speeches: {len(df_final):,}")
    print(f"Unique speakers: {df_final[SPEAKER_ID_COLUMN].nunique():,}")
    print(f"Mean ideology score: {df_final[TARGET_COLUMN].mean():.3f}")
    print(f"Ideology score range: [{df_final[TARGET_COLUMN].min():.3f}, "
          f"{df_final[TARGET_COLUMN].max():.3f}]")
    print(f"\nOutput file: {CLEANED_SPEECHES}")
    print("\n" + "="*80)
    print("STAGE 1 COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
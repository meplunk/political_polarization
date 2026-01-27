"""
===============================================================================
FILE: stage_01_clean.py
AUTHOR: Mary Edith Plunkett
PROJECT: Political Polarization Project
DATE: January 27, 2026
===============================================================================
PURPOSE:
    Clean and merge Congressional speeches and ad transcripts with DIME 
    ideology scores. This is Stage 1 of the NLP pipeline.

DESCRIPTION:
    1. Filter DIME recipients to federal candidates (2013-2018)
    2. Merge Congressional speeches with speaker metadata
    3. Link speeches to DIME ideology scores
    4. Load ad transcripts and apply WMP cleaning filters
    5. Link ads to DIME scores
    6. Create two ad files:
       - All airings (for analysis of ad volume/timing)
       - Unique ads (for text analysis/tokenization)

INPUT FILES:
    - data/01_raw/dime_recipients_1979_2024.csv
    - data/01_raw/speeches_114.txt
    - data/01_raw/114_SpeakerMap.txt
    - data/01_raw/2016HouseVideo/*.txt (ad transcripts)
    - data/01_raw/wmp-house-2016-v1.0.dta (ad metadata)

OUTPUT FILES:
    - data/02_cleaned/cleaned_speeches.csv
    - data/02_cleaned/cleaned_ads_airings.csv
    - data/02_cleaned/cleaned_ads_unique.csv

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
    df['unique_id'] = (
        df['lname'].str.split().str[0].str.upper() + '_' +
        df['distcyc'].str.split('_', n=1).str[1]
    )
    
    return df


def merge_speakers_and_speeches(speaker_path, speech_path):
    """
    Merge Congressional Record speaker metadata with speech texts.
    """
    print("Loading Congressional Record data...")
    
    df_speakers = pd.read_csv(speaker_path, delimiter="|", encoding="latin1")
    df_speeches = pd.read_csv(speech_path, delimiter="|", encoding="latin1",
                              on_bad_lines="warn")
    
    print(f"  Loaded {len(df_speakers):,} speakers")
    print(f"  Loaded {len(df_speeches):,} speeches")
    
    df_merged = pd.merge(df_speakers, df_speeches, on="speech_id", how="right")
    
    keep_cols = ["speakerid", "speech_id", "lastname", "firstname",
                 "speech", "state", "district"]
    df_merged = df_merged[keep_cols]
    
    df_merged = df_merged[df_merged['lastname'].notna() & 
                          (df_merged['lastname'] != '')]
    
    df_merged['unique_id'] = (
        df_merged['lastname'].str.split().str[0].str.upper() + '_' +
        df_merged['state'].astype(str).str.upper() + '_' +
        df_merged['district'].apply(lambda x: str(int(x)) if pd.notnull(x) else 'S')
    )
    
    df_merged = apply_name_corrections(df_merged)
    
    print(f"  Merged dataset: {len(df_merged):,} speeches")
    
    return df_merged


def apply_name_corrections(df):
    """Apply manual corrections for known naming mismatches."""
    corrections = {
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
        "VAN_MD_8": "VANHOLLEN_MD_8",
        "ROS-LEHTINEN_FL_27": "ROS_FL_27",
        "DIAZ-BALART_FL_25": "DIAZ_FL_25",
        "ROYBAL-ALLARD_CA_40": "ROYBAL_CA_40",
        "JACKSON_TX_18": "LEE_TX_18"
    }
    
    df['unique_id'] = df['unique_id'].replace(corrections)
    return df


def merge_with_ideology_scores(df_speeches, df_dime):
    """Merge speeches with DIME ideology scores."""
    print("Merging speeches with ideology scores...")
    
    merged = pd.merge(df_speeches, df_dime, on='unique_id', how='left')
    
    column_mapping = {
        'speech': TEXT_COLUMN,
        'dwdime': TARGET_COLUMN
    }
    merged = merged.rename(columns=column_mapping)
    
    keep_cols = [
        'speakerid',
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
    
    keep_cols = [col for col in keep_cols if col in merged.columns]
    merged = merged[keep_cols]
    
    initial_count = len(merged)
    merged = merged.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN])
    print(f"  Dropped {initial_count - len(merged):,} rows with missing text or scores")
    
    if MIN_SPEECH_LENGTH > 0:
        merged = merged[merged[TEXT_COLUMN].str.len() >= MIN_SPEECH_LENGTH]
        print(f"  Kept speeches with length >= {MIN_SPEECH_LENGTH} characters")
    
    merged = merged.reset_index(drop=True)
    print(f"  Final dataset: {len(merged):,} speeches")
    
    return merged


def load_and_clean_ads(video_dir, metadata_path, df_dime):
    """
    Load ad transcripts and merge with metadata and DIME scores.
    
    Applies WMP cleaning logic (Anderson, Ciliberto, Leyden 2025).
    """
    print("Loading ad transcripts...")
    
    txt_files = list(Path(video_dir).glob("*.txt"))
    
    if not txt_files:
        print(f"  Warning: No .txt files found in {video_dir}")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"  Found {len(txt_files):,} ad transcript files")
    
    ads_data = []
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            ads_data.append({
                AD_ID_COLUMN: txt_file.stem,
                AD_TEXT_COLUMN: text
            })
    
    df_ads_text = pd.DataFrame(ads_data)
    print(f"  Loaded {len(df_ads_text):,} unique ad transcripts")
    
    # Load and clean WMP metadata
    print("Loading and cleaning WMP metadata...")
    df = pd.read_stata(metadata_path, convert_categoricals=False)
    print(f"  Loaded {len(df):,} ad airings")
    
    # Apply Anderson, Ciliberto, Leyden (2025) cleaning
    print("  Applying data filters...")
    
    df['state'] = df['categorystate']
    
    # Restrict to general election
    df = df[df['election'] == 'GENERAL']
    print(f"    After general election filter: {len(df):,}")
    
    # Restrict to major affiliates
    df = df[df['affiliate'].isin(['ABC', 'CBS', 'CW', 'FOX', 'NBC'])]
    print(f"    After major affiliate filter: {len(df):,}")
    
    # Drop low-cost ads (< 5th percentile)
    est_cost_5pct = df['est_cost'].quantile(0.05)
    df = df[df['est_cost'] > est_cost_5pct]
    print(f"    After cost filter (>${est_cost_5pct:.0f}): {len(df):,}")
    
    # Set senate district to 60
    df.loc[df['senate'] == 1, 'district'] = '60'
    
    # Create time variables
    df['year'] = '2016'
    df['race_id'] = df['state'] + df['district'].astype(str) + '-' + df['year']
    
    election_day = pd.to_datetime('2016-11-08')
    df['day'] = (election_day - df['airdate']).dt.days
    df['week'] = df['day'] // 7
    df['biweek'] = df['day'] // 14
    
    # Filter by issue coding
    df = df[df['codingstatus'] == 1]
    print(f"    After issue coding filter: {len(df):,}")
    
    # Get issue variables
    issue_vars = [col for col in df.columns if 'issue' in col.lower() and 
                  col not in ['issue', 'issue96', 'issue97', 'issue97_txt', 'codingstatus']]
    
    # Correct issue coding (anything > 0 becomes 1)
    for var in issue_vars:
        if var in df.columns:
            df[var] = df[var].apply(lambda x: 1 if pd.notnull(x) and x > 0 else 0)
    
    # Keep only ads with at least one coded issue
    df['n_coded_issues'] = df[issue_vars].sum(axis=1)
    df = df[df['n_coded_issues'] > 0]
    df = df.drop(columns=['n_coded_issues'])
    print(f"    After requiring coded issues: {len(df):,}")
    
    # Clean candidate IDs
    df = df[df['cand_id'].notna()]
    df = df[~df['cand_id'].astype(str).str.match(r'^\s*$')]
    print(f"    After cleaning cand_id: {len(df):,}")
    
    # Create unique_id
    def make_unique_id(row):
        pattern = r"[_,\- ]"
        last = re.split(pattern, str(row['cand_id']))[0]
        last = last.replace("'", "").upper()
        district = re.sub(r"^0+", "", str(row['district'])) or "0"
        return f"{last}_{row['state']}_{district}"
    
    df['unique_id'] = df.apply(make_unique_id, axis=1)
    
    # Apply manual corrections
    ad_corrections = {
        "LINDBECK_AK_1": "LINDBECK_AK_NA",
        "DAVID_IA_3": "YOUNG_IA_3",
        "ANDY_KY_6": "BARR_KY_6",
        "BEUTLER_WA_3": "HERRERA_WA_3",
        "RODGERS_WA_5": "MCMORRIS_WA_5"
    }
    df['unique_id'] = df['unique_id'].replace(ad_corrections)
    
    # Merge with DIME scores (deduplicated)
    print("  Merging with DIME scores...")
    df_dime_dedup = df_dime.sort_values('cycle', ascending=False).drop_duplicates(
        subset=['unique_id'], keep='first'
    )
    
    df = pd.merge(df, df_dime_dedup[['unique_id', 'dwdime']], 
                  on='unique_id', how='left')
    df = df.rename(columns={'dwdime': TARGET_COLUMN})
    
    # Create AIRINGS dataset
    airings_cols = [AD_ID_COLUMN, 'unique_id', TARGET_COLUMN, 'party', 
                    'race_id', 'biweek']
    airings_cols = [col for col in airings_cols if col in df.columns]
    df_airings = df[airings_cols].copy()
    
    initial_airings = len(df_airings)
    df_airings = df_airings.dropna(subset=[TARGET_COLUMN])
    print(f"  Airings: kept {len(df_airings):,} with scores (dropped {initial_airings - len(df_airings):,})")
    
    # Create UNIQUE ADS dataset
    df_unique = df.drop_duplicates(subset=[AD_ID_COLUMN], keep='first')
    df_unique = pd.merge(df_ads_text, df_unique, on=AD_ID_COLUMN, how='left')
    
    unique_cols = [AD_ID_COLUMN, AD_TEXT_COLUMN, 'unique_id', TARGET_COLUMN, 'party']
    unique_cols = [col for col in unique_cols if col in df_unique.columns]
    df_unique = df_unique[unique_cols]
    
    initial_unique = len(df_unique)
    df_unique = df_unique.dropna(subset=[AD_TEXT_COLUMN, TARGET_COLUMN])
    
    if MIN_AD_LENGTH > 0:
        df_unique = df_unique[df_unique[AD_TEXT_COLUMN].str.len() >= MIN_AD_LENGTH]
    
    print(f"  Unique ads: kept {len(df_unique):,} with text and scores (dropped {initial_unique - len(df_unique):,})")
    
    df_airings = df_airings.reset_index(drop=True)
    df_unique = df_unique.reset_index(drop=True)
    
    return df_airings, df_unique


def main():
    """Execute the complete Stage 1 cleaning pipeline."""
    print("\n" + "="*80)
    print("STAGE 1: DATA CLEANING")
    print("="*80 + "\n")
    
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
    n_speakers = df_speeches_final['speakerid'].nunique()
    print(f"  Unique speakers: {n_speakers:,}")
    mean_score = df_speeches_final[TARGET_COLUMN].mean()
    print(f"  Mean ideology score: {mean_score:.3f}")
    
    if len(df_ads_airings) > 0:
        print(f"\nAD AIRINGS:")
        print(f"  Total airings: {len(df_ads_airings):,}")
        n_unique_ads = df_ads_airings[AD_ID_COLUMN].nunique()
        print(f"  Unique ads: {n_unique_ads:,}")
        mean_score = df_ads_airings[TARGET_COLUMN].mean()
        print(f"  Mean ideology score: {mean_score:.3f}")
    
    if len(df_ads_unique) > 0:
        print(f"\nUNIQUE ADS (for tokenization):")
        print(f"  Total unique ads: {len(df_ads_unique):,}")
        n_candidates = df_ads_unique['unique_id'].nunique()
        print(f"  Unique candidates: {n_candidates:,}")
        mean_score = df_ads_unique[TARGET_COLUMN].mean()
        print(f"  Mean ideology score: {mean_score:.3f}")
    
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

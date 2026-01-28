"""
===============================================================================
FILE: stage_03a_tfidf.py
AUTHOR: Mary Edith Plunkett
PROJECT: Political Polarization Project (PPP)
DATE: January 27, 2026
===============================================================================
PURPOSE:
    This script performs the third stage of the data processing pipeline:
    cross-validation to find optimal TF-IDF and Ridge regression parameters
    for predicting political ideology scores from speech text.

DESCRIPTION OF STEPS:
    1. Load tokenized ads and speeches from Stage 2 (stage_02_tokenize.py)
    2. Define parameter grid for testing (min_df, max_df, ridge_alpha)
    3. For each parameter combination:
       - Fit TF-IDF vectorizer on ad transcripts to learn vocabulary
       - Transform speeches using learned vocabulary
       - Train Ridge regression model to predict ideology scores
       - Evaluate using cross-validation
    4. Identify best parameter combination based on CV R² score
    5. Train final model on full dataset using best parameters
    6. Save model, vectorizer, and results for downstream use

INPUT FILES:
    - data/03_tokens/tokenized_ads_unique.csv       (from stage_02_tokenize.py)
    - data/03_tokens/agg_tokenized_speeches.pkl     (from stage_02_tokenize.py + aggregation helper)

OUTPUT FILES:
    - models/tfidf_cv_results.pkl              (all parameter combinations tested)
    - models/tfidf_model.pkl              (final trained Ridge model)
    - models/tfidf_vectorizer.pkl         (final trained TF-IDF vectorizer)

DEPENDENCIES:
    - scikit-learn (TfidfVectorizer, Ridge, cross-validation)
    - numpy, pandas
    - utils.py (project path management)

USAGE:
    Run directly from the command line or import and call `main()`:
        $ python n3_tfidf_model.py

===============================================================================
"""
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from config import (TFIDF_MODEL, load_pickle, save_pickle, TOKENIZED_ADS, 
                    AGG_TOKENIZED_SPEECHES, TFIDF_CV_RESULTS, TFIDF_VECTORIZER)
import helper_aggregate_by_speaker as agg_helper
import sys
from pathlib import Path
from tqdm import tqdm
import os

def identity(x):
    return x


def load_tokenized_data():
    """
    Load tokenized ads and speeches from Stage 2 outputs.

    This function loads the pre-tokenized text data and performs basic
    validation to ensure required columns exist and data is clean.

    Returns:
        tuple: (ads_tokens, speeches_tokens, speeches_df)
            - ads_tokens: List of token lists for ads (for vocabulary fitting)
            - speeches_tokens: List of token lists for speeches (for prediction)
            - speeches_df: Full speeches DataFrame with metadata and targets

    Raises:
        FileNotFoundError: If tokenized data files don't exist
        ValueError: If required columns are missing
    """
    print("\n=== Loading Tokenized Data ===")

    ads_df = pd.read_csv(TOKENIZED_ADS)

    print(f"Loaded {len(ads_df):,} tokenized ads")

    agg_path = Path(AGG_TOKENIZED_SPEECHES)

    if not agg_path.exists():
        agg_helper.main()

    speeches_df = load_pickle(agg_path)
    print(f"Loaded {len(speeches_df):,} speakers with speech tokens and metadata")

    ads_token_col = "tokenized_ad"
    speech_token_col = "speech"
    target_col = "dime"

    # --- Clean data: remove rows with missing values ---
    before = len(speeches_df)
    speeches_df = speeches_df.dropna(subset=[speech_token_col, target_col]).reset_index(drop=True)
    removed = before - len(speeches_df)
    if removed > 0:
        print(f"Removed {removed:,} rows with missing tokens or ideology scores")

    # --- Extract token lists from DataFrames ---
    ads_tokens = ads_df[ads_token_col].tolist()
    speeches_tokens = speeches_df[speech_token_col].tolist()

    return ads_tokens, speeches_tokens, speeches_df

def define_parameter_grid():
    """
    Define the TF-IDF parameter grid (min_df, max_df, ngram).
    Ridge alphas are handled separately so we can reuse X across alphas.
    """
    min_df_values = [2, 5, 10, 100]
    max_df_values = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    ngram_values = [1, 2]

    param_combinations = []
    for min_df in min_df_values:
        for max_df in max_df_values:
            for ngram in ngram_values:
                param_combinations.append({
                    'min_df': min_df,
                    'max_df': max_df,
                    'ngram': ngram,
                })

    print("\n=== TF-IDF Parameter Grid ===")
    print(f"Total TF-IDF combinations to test: {len(param_combinations)}")
    print(f"  min_df values: {min_df_values}")
    print(f"  max_df values: {max_df_values}")
    print(f"  N-grams: {ngram_values}")
    print(f"  Total: {len(min_df_values)} × {len(max_df_values)} × {len(ngram_values)}"
          f" = {len(param_combinations)}")

    # alpha grid is separate
    ridge_alpha_values = [0.1, 1, 10, 100, 1000]

    return param_combinations, ridge_alpha_values


def create_tfidf_vectorizer(min_df, max_df, ngram):
    """
    Create and configure a TF-IDF vectorizer.

    TF-IDF (Term Frequency - Inverse Document Frequency) vectorization:
    - Downweights common terms that appear in many documents
    - Upweights distinctive terms that appear in few documents
    - Better for capturing ideological signal than raw counts

    Args:
        min_df (int): Minimum document frequency (ignore rarer terms)
        max_df (float): Maximum document frequency (ignore common terms)

    Returns:
        TfidfVectorizer: Configured vectorizer instance
    """
    return TfidfVectorizer(
        ngram_range=(1, ngram),  # Use unigrams only (single words)
        preprocessor=identity,  # No preprocessing (already tokenized)
        tokenizer=identity,  # No tokenization (already tokenized)
        token_pattern=None,  # Disable regex tokenization
        min_df=min_df,  # Minimum document frequency
        max_df=max_df  # Maximum document frequency (as proportion)
    )


def evaluate_ridge_for_alpha(X, target, base_params, alpha, vocab_size, cv=5):
    """
    Evaluate a Ridge model for a given alpha using precomputed TF-IDF matrix X.

    base_params: dict with min_df, max_df, ngram
    """
    start_time = time.time()

    try:
        ridge = Ridge(alpha=alpha)

        # Compute R² and MSE in a *single* cross-validated run
        scoring = {
            "r2": "r2",
            "neg_mse": "neg_mean_squared_error",
        }

        N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

        cv_results = cross_validate(
            ridge,
            X,
            target,
            cv=cv,
            scoring=scoring,
            n_jobs=N_JOBS,
            return_estimator=False,
        )

        r2_scores = cv_results["test_r2"]
        mse_scores = -cv_results["test_neg_mse"]  # flip sign back

        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        mean_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)

        # Fit on full data for train R²
        ridge.fit(X, target)
        train_r2 = r2_score(target, ridge.predict(X))

        results = {
            "min_df": base_params["min_df"],
            "max_df": base_params["max_df"],
            "ridge_alpha": alpha,
            "max_ngram": base_params["ngram"],
            "vocab_size": vocab_size,
            "cv_r2_mean": mean_r2,
            "cv_r2_std": std_r2,
            "cv_mse_mean": mean_mse,
            "cv_mse_std": std_mse,
            "train_r2": train_r2,
            "evaluation_time": time.time() - start_time,
            "success": True,
        }

    except Exception as e:
        results = {
            "min_df": base_params["min_df"],
            "max_df": base_params["max_df"],
            "ridge_alpha": alpha,
            "max_ngram": base_params["ngram"],
            "vocab_size": vocab_size,
            "cv_r2_mean": -999,
            "cv_r2_std": 0,
            "cv_mse_mean": 999,
            "cv_mse_std": 0,
            "train_r2": -999,
            "evaluation_time": time.time() - start_time,
            "success": False,
            "error": str(e),
        }

    return results

def run_cross_validation():

    print("\n=== Running Cross-Validation ===")

    # Load data
    ads_tokens, speeches_tokens, speeches_df = load_tokenized_data()
    target = speeches_df["dime"]

    # Define TF-IDF and alpha grids
    tfidf_param_combinations, ridge_alpha_values = define_parameter_grid()

    print("\n=== Evaluating Parameter Combinations ===")
    results = []
    total_tfidf_combos = len(tfidf_param_combinations)
    min_vocab = 50  # same threshold you used before

    combo_counter = 0
    for tfidf_idx, params in enumerate(
        tqdm(
            tfidf_param_combinations,
            total=total_tfidf_combos,
            desc="TF-IDF configs",
            file=sys.stdout,
            leave=True,
        ),
        1,
    ):
        combo_counter += 1

        print(f"\n[TFIDF {tfidf_idx}/{total_tfidf_combos}]")
        print(f"  Min DF: {params['min_df']}, Max DF: {params['max_df']}, Ngrams: {params['ngram']}")

        # --- Step 1: create and fit vectorizer once for this TF-IDF config ---
        start_tfidf = time.time()
        vectorizer = create_tfidf_vectorizer(
            params["min_df"],
            params["max_df"],
            params["ngram"],
        )

        try:
            vectorizer.fit(ads_tokens)
            vocab_size = len(vectorizer.vocabulary_)

            if vocab_size < min_vocab:
                raise ValueError(
                    f"Vocabulary too small (size={vocab_size} < min_vocab={min_vocab}) "
                    f"for min_df={params['min_df']}, max_df={params['max_df']}"
                )

            # Transform speeches once
            X = vectorizer.transform(speeches_tokens)
            # Optional: downcast to float32 to save memory
            # X = X.astype("float32")

            tfidf_time = time.time() - start_tfidf
            print(f"  TF-IDF prepared: vocab={vocab_size:,}, "
                  f"matrix={X.shape[0]:,}×{X.shape[1]:,}, time={tfidf_time:.2f}s")

            # --- Step 2: loop over alphas using the same X ---
            for alpha in tqdm(
                ridge_alpha_values,
                desc="Ridge alphas",
                file=sys.stdout,
                leave=False,
            ):
                print(f"    [alpha={alpha}]")
                result = evaluate_ridge_for_alpha(
                    X,
                    target,
                    base_params=params,
                    alpha=alpha,
                    vocab_size=vocab_size,
                    cv=5,
                )
                results.append(result)

                if result["success"]:
                    print(f"      ✓ CV R²: {result['cv_r2_mean']:.4f} (±{result['cv_r2_std']:.4f})")
                    print(f"        CV MSE: {result['cv_mse_mean']:.4f} (±{result['cv_mse_std']:.4f})")
                    print(f"        Train R²: {result['train_r2']:.4f}")
                else:
                    print(f"      ✗ FAILED: {result.get('error', 'Unknown error')}")
                print(f"        Time (Ridge only): {result['evaluation_time']:.2f}s")

        except Exception as e:
            # If TF-IDF itself fails, mark all alphas as failed for this combo
            tfidf_time = time.time() - start_tfidf
            print(f"  ✗ TF-IDF FAILED for this config: {e} (after {tfidf_time:.2f}s)")

            for alpha in ridge_alpha_values:
                results.append({
                    "min_df": params["min_df"],
                    "max_df": params["max_df"],
                    "ridge_alpha": alpha,
                    "max_ngram": params["ngram"],
                    "vocab_size": 0,
                    "cv_r2_mean": -999,
                    "cv_r2_std": 0,
                    "cv_mse_mean": 999,
                    "cv_mse_std": 0,
                    "train_r2": -999,
                    "evaluation_time": tfidf_time,
                    "success": False,
                    "error": f"TF-IDF failure: {e}",
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save all results
    results_df.to_csv(TFIDF_CV_RESULTS, index=False)
    print(f"\n✓ Saved all results to {TFIDF_CV_RESULTS}")

    # === Find best parameters (same as before) ===
    successful_results = results_df[results_df["success"] == True]
    if len(successful_results) == 0:
        raise ValueError("No successful parameter combinations found! Check data and parameters.")

    best_idx = successful_results["cv_r2_mean"].idxmax()
    best_params = successful_results.loc[best_idx].to_dict()

    print(f"\n=== Best Parameters Found ===")
    print(f"CV R²: {best_params['cv_r2_mean']:.4f} (±{best_params['cv_r2_std']:.4f})")
    print(f"CV MSE: {best_params['cv_mse_mean']:.4f} (±{best_params['cv_mse_std']:.4f})")
    print(f"Train R²: {best_params['train_r2']:.4f}")
    print(f"  Min DF: {best_params['min_df']}")
    print(f"  Max DF: {best_params['max_df']}")
    print(f"  Ridge Alpha: {best_params['ridge_alpha']}")
    print(f"  Vocabulary size: {best_params['vocab_size']:,}")

    # === Train final model using the best TF-IDF + alpha ===
    print("\n=== Training Final Model ===")
    print("Using best parameters on full dataset...")

    best_vectorizer = create_tfidf_vectorizer(
        best_params["min_df"],
        best_params["max_df"],
        best_params["max_ngram"],   # <-- bugfix: you were missing ngram here
    )

    best_vectorizer.fit(ads_tokens)
    X_best = best_vectorizer.transform(speeches_tokens)
    print(f"Final feature matrix: {X_best.shape[0]:,} speeches × {X_best.shape[1]:,} features")

    best_model = Ridge(alpha=best_params["ridge_alpha"])
    best_model.fit(X_best, target)

    save_pickle(best_model, TFIDF_MODEL)
    save_pickle(best_vectorizer, TFIDF_VECTORIZER)

    print(f"✓ Saved best model to {paths['models']/'best_agg_tfidf_model.pkl'}")
    print(f"✓ Saved best vectorizer to {paths['vectorizers']/'best_agg_tfidf_vectorizer.pkl'}")

    return results_df, best_params, best_model


def analyze_results(results_df):
    """
    Analyze and display comprehensive results from cross-validation.

    This function provides multiple views of the CV results:
    - Overall success/failure statistics
    - Top 5 best parameter combinations
    - Performance by min_df (vocabulary size control)
    - Performance by max_df (common term filtering)
    - Performance by Ridge alpha (regularization strength)

    Args:
        results_df (pd.DataFrame): Results from cross-validation
    """
    successful_results = results_df[results_df['success'] == True]

    print(f"\n{'=' * 60}")
    print(f"{'CROSS-VALIDATION ANALYSIS':^60}")
    print(f"{'=' * 60}")

    print(f"\nOverall Statistics:")
    print(f"  Total combinations tested: {len(results_df)}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(results_df) - len(successful_results)}")

    if len(successful_results) == 0:
        print("\n⚠ No successful combinations to analyze!")
        return

    # Top 5 results
    print(f"\n{'=' * 60}")
    print(f"TOP 5 PARAMETER COMBINATIONS (by CV R²)")
    print(f"{'=' * 60}")

    top_5 = successful_results.nlargest(5, 'cv_r2_mean')

    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"\n{idx}. CV R²: {row['cv_r2_mean']:.4f} (±{row['cv_r2_std']:.4f})")
        print(f"   CV MSE: {row['cv_mse_mean']:.4f} (±{row['cv_mse_std']:.4f})")
        print(f"   Train R²: {row['train_r2']:.4f}")
        print(f"   Min DF: {row['min_df']}, Max DF: {row['max_df']}, Ridge Alpha: {row['ridge_alpha']}")
        print(f"   Vocabulary: {row['vocab_size']:,} terms")

    # Analysis by min_df
    print(f"\n{'=' * 60}")
    print(f"PERFORMANCE BY MIN_DF (Minimum Document Frequency)")
    print(f"{'=' * 60}")

    for min_df in sorted(successful_results['min_df'].unique()):
        subset = successful_results[successful_results['min_df'] == min_df]
        mean_r2 = subset['cv_r2_mean'].mean()
        max_r2 = subset['cv_r2_mean'].max()
        mean_vocab = subset['vocab_size'].mean()
        print(
            f"Min DF = {min_df:3d}: Mean R² = {mean_r2:.4f}, Max R² = {max_r2:.4f}, Avg Vocab = {mean_vocab:,.0f} ({len(subset)} combos)")

    # Analysis by max_df
    print(f"\n{'=' * 60}")
    print(f"PERFORMANCE BY MAX_DF (Maximum Document Frequency)")
    print(f"{'=' * 60}")

    for max_df in sorted(successful_results['max_df'].unique()):
        subset = successful_results[successful_results['max_df'] == max_df]
        mean_r2 = subset['cv_r2_mean'].mean()
        max_r2 = subset['cv_r2_mean'].max()
        mean_vocab = subset['vocab_size'].mean()
        print(
            f"Max DF = {max_df:.2f}: Mean R² = {mean_r2:.4f}, Max R² = {max_r2:.4f}, Avg Vocab = {mean_vocab:,.0f} ({len(subset)} combos)")

    # Analysis by Ridge alpha
    print(f"\n{'=' * 60}")
    print(f"PERFORMANCE BY RIDGE ALPHA (Regularization Strength)")
    print(f"{'=' * 60}")

    for alpha in sorted(successful_results['ridge_alpha'].unique()):
        subset = successful_results[successful_results['ridge_alpha'] == alpha]
        mean_r2 = subset['cv_r2_mean'].mean()
        max_r2 = subset['cv_r2_mean'].max()
        print(f"Alpha = {alpha:6.1f}: Mean R² = {mean_r2:.4f}, Max R² = {max_r2:.4f} ({len(subset)} combos)")

    print(f"\n{'=' * 60}\n")


def main():
    """
    Main execution function for standalone runs.

    This function orchestrates the complete model training pipeline:
    1. Run cross-validation
    2. Analyze and display results
    3. Report final model performance
    """
    print("\n" + "=" * 60)
    print("STAGE 4a: TF-IDF MODEL TRAINING & CROSS-VALIDATION")
    print("=" * 60)

    start_time = time.time()

    # Run cross-validation
    results_df, best_params, best_model = run_cross_validation()

    # Analyze results
    analyze_results(results_df)

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"{'PIPELINE COMPLETE':^60}")
    print(f"{'=' * 60}")
    print(f"\nTotal execution time: {total_time / 60:.2f} minutes ({total_time:.0f} seconds)")
    print(f"Best CV R²: {best_params['cv_r2_mean']:.4f} (±{best_params['cv_r2_std']:.4f})")
    print(f"Best CV MSE: {best_params['cv_mse_mean']:.4f} (±{best_params['cv_mse_std']:.4f})")
    print(f"\nBest configuration:")
    print(f"  Min DF: {best_params['min_df']}")
    print(f"  Max DF: {best_params['max_df']}")
    print(f"  Ridge Alpha: {best_params['ridge_alpha']}")
    print(f"  Vocabulary size: {best_params['vocab_size']:,} terms")
    print(f"\n✓ Model and vectorizer saved for downstream use")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
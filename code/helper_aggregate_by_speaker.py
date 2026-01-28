import pandas as pd
from config import (load_pickle, save_pickle, TOKENIZED_SPEECHES, AGG_TOKENIZED_SPEECHES)
from itertools import chain

def aggregate_by_speaker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates a DataFrame by `_unique_id`:
      - Drops `_speakerid`, `_speech_id`
      - Assumes `_gen.vote.pct`, `_gwinner`, `_dwdime` are constant within each `_unique_id`
      - Concatenates `_speech_tokens` (lists of tokens) for rows with the same `_unique_id`
    """
    cols_to_drop = ['speakerid', 'speech_id']
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop)

    agg_dict = {
        'gen.vote.pct': 'first',
        'gwinner': 'first',
        'dwdime': 'first',
        'tokenized_speech': lambda x: list(chain.from_iterable(x))
    }

    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    grouped = (
        df
        .groupby('unique_id', as_index=False)
        .agg(agg_dict)
    )

    return grouped

def main():
    print("Loading tokenized speeches...")
    df = pd.read_csv(TOKENIZED_SPEECHES)
    print(df.columns)
    print("Aggregating by speaker...")
    df_aggregated = aggregate_by_speaker(df)
    print("Saving aggregated DataFrame...")
    df_aggregated.to_csv(AGG_TOKENIZED_SPEECHES, index=False)

if __name__ == "__main__":
    main()
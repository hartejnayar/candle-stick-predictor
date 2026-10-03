import pandas as pd
import numpy as np
import logging
import pathlib 
import yaml


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_and_clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans raw OHLCV financial time-series data.
    """
    initial_count = len(df)
    if initial_count == 0:
        logging.warning("The input DataFrame is empty.")
        return df

    # deduplicate by timestamp
    df = df.drop_duplicates(subset= ['timestamp']).copy()
    dedup_count = len(df)

    if initial_count - dedup_count > 0:
        logging.info(f"Removed {initial_count - dedup_count} duplicate rows based on timestamp.")

    # sorting choronologically
    df = df.sort_values('timestamp').reset_index(drop=True)
    logging.info("Data sorted chronologically by timestamp.")

    # check for missing values
    null_counts = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum().sum()
    if null_counts > 0:
        logging.warning(f"Found {null_counts} missing values in OHLCV columns. Dropping rows with missing values.")
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)
        logging.info(f"Rows with missing values dropped. Remaining rows: {len(df)}.")

    # enforce invalid OHLC relationship checks
    invalid_high = (df['high'] < df['open']) | (df['high'] < df['close'])
    invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])
    invalid_prices = (df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)

    invalid_mask = invalid_high | invalid_low | invalid_prices
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        logging.warning(f"Found {invalid_count} invalid OHLCV rows. Dropping them.")
        df = df[~invalid_mask].reset_index(drop=True)
        logging.info(f"Invalid rows dropped. Remaining rows: {len(df)}.")

    return df

def preprocess_all_raw_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """Reads all raw CSVs, cleans them, and writes to processed directory."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    for file_path in raw_path.glob("*.csv"):
        logging.info(f"Processing {file_path.name}...")
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        cleaned_df = validate_and_clean_ohlcv(df)
        
        out_path = processed_path / file_path.name
        cleaned_df.to_csv(out_path, index=False)
        logging.info(f"Saved cleaned dataset to {out_path}")

if __name__ == "__main__":
    preprocess_all_raw_data()
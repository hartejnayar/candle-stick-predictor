import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OHLCVPreprocessor:
    """Cleans, normalizes, and prepares raw OHLCV time-series data."""
    
    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates required columns, handles missing values, removes duplicates, 
        and ensures chronological sorting.
        """
        df = df.copy()
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in preprocessor: {col}")
                
        # Drop rows with missing values or invalid prices
        df = df.dropna(subset=required_cols)
        df = df[(df['high'] >= df['low']) & (df['open'] > 0) & (df['close'] > 0)]
        
        # Sort chronologically and remove duplicate timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        logging.info(f"Successfully cleaned OHLCV data. Total valid rows: {len(df)}")
        return df

def validate_and_clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience helper function matching calls across the project."""
    return OHLCVPreprocessor.validate_and_clean(df)
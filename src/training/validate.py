import pandas as pd
import numpy as np
from typing import Tuple

def chronological_train_test_split(
    df: pd.DataFrame, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a time-ordered DataFrame chronologically into train, validation, and test sets.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df
import pandas as pd
from src.data.preprocessing import validate_and_clean_ohlcv

def test_data_preprocessing():
    df = pd.DataFrame({
        'timestamp': ['2026-01-01', '2026-01-02'],
        'open': [100, 102],
        'high': [105, 106],
        'low': [98, 100],
        'close': [102, 104],
        'volume': [1000, 1200]
    })
    cleaned = validate_and_clean_ohlcv(df)
    assert len(cleaned) == 2
import pandas as pd
from src.features.technical_indicators import TechnicalIndicators

def test_technical_indicators():
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2026-01-01', periods=30),
        'open': [100]*30, 'high': [105]*30, 'low': [95]*30,
        'close': range(100, 130), 'volume': [1000]*30
    })
    result = TechnicalIndicators.add_all_indicators(df)
    assert 'rsi' in result.columns
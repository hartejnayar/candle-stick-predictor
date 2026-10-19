import pandas as pd
import numpy as np
from src.patterns.candlestick_patterns import PatternDetector

def test_morning_star_detection():
    # 3-candle sequence: Long Bearish, Small Star, Strong Bullish
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=3),
        'open': [100.0, 93.0, 94.0],
        'high': [101.0, 94.5, 99.0],
        'low': [92.0, 92.5, 93.0],
        'close': [93.0, 93.5, 98.0],
        'volume': [1000, 800, 1500]
    }
    df = pd.DataFrame(data)
    is_morning_star = PatternDetector.detect_morning_star(df).iloc[2]
    assert is_morning_star == True

def test_marubozu_detection():
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=1),
        'open': [100.0],
        'high': [105.0],
        'low': [100.0],
        'close': [105.0],
        'volume': [2000]
    }
    df = pd.DataFrame(data)
    is_marubozu = PatternDetector.detect_marubozu_bullish(df).iloc[0]
    assert is_marubozu == True
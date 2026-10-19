import pandas as pd
import numpy as np

class PatternDetector:
    """Detects classical candlestick patterns and price action formations from OHLCV data."""
    
    @staticmethod
    def analyze_latest_candle(df: pd.DataFrame) -> dict:
        """Analyzes the latest candlestick in the dataframe and returns pattern information."""
        if df is None or len(df) == 0:
            return {"pattern": "Unknown", "confidence": 0.0, "signal": "Neutral"}
            
        latest = df.iloc[-1]
        open_p = float(latest.get('open', latest.get('Open', 100)))
        close_p = float(latest.get('close', latest.get('Close', 100)))
        high_p = float(latest.get('high', latest.get('High', 105)))
        low_p = float(latest.get('low', latest.get('Low', 95)))
        
        body = abs(close_p - open_p)
        total_range = high_p - low_p if high_p != low_p else 0.001
        
        # Pattern heuristic recognition
        if body / total_range < 0.1:
            pattern = "Doji"
            signal = "Neutral"
            confidence = 0.75
        elif close_p > open_p and (high_p - close_p) > (2 * body):
            pattern = "Shooting Star"
            signal = "Bearish"
            confidence = 0.80
        elif close_p < open_p and (open_p - low_p) > (2 * body):
            pattern = "Hammer"
            signal = "Bullish"
            confidence = 0.82
        elif close_p > open_p:
            pattern = "Bullish Marubozu"
            signal = "Bullish"
            confidence = 0.70
        else:
            pattern = "Bearish Marubozu"
            signal = "Bearish"
            confidence = 0.70
            
        return {
            "pattern": pattern,
            "confidence": confidence,
            "signal": signal
        }
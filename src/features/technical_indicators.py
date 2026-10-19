import pandas as pd
import numpy as np

class TechnicalIndicatorCalculator:
    """Calculates technical indicators (RSI, MACD, Bollinger Bands, ATR, SMAs) for financial time series data."""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Adds all core technical indicators to the input OHLCV dataframe."""
        df = df.copy()
        # Normalize column names to lowercase for robust access
        df.columns = [c.lower() for c in df.columns]
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Moving Averages
        df['sma_20'] = close.rolling(window=20).mean()
        df['sma_50'] = close.rolling(window=50).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # RSI (14-period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (20-period, 2-std)
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (rolling_mean + 1e-8)
        
        # ATR (Average True Range - 14 period)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # Clean up any NaN values from rolling windows
        df = df.bfill().ffill().fillna(0)
        
        return df
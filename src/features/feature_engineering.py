import pandas as pd
import numpy as np
import ta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    """Extracts technical indicators and constructs normalized feature matrices."""

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['sma_10'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_10'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
        
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        
        return df.dropna().reset_index(drop=True)

    @classmethod
    def extract_feature_matrix(cls, df: pd.DataFrame):
        df_feat = cls.extract_features(df)
        feature_cols = [
            'rsi', 'macd', 'macd_signal', 'sma_10', 'sma_50', 
            'ema_10', 'bb_high', 'bb_low', 'bb_width', 'atr', 'volume_ratio'
        ]
        available_cols = [c for c in feature_cols if c in df_feat.columns]
        feature_matrix = df_feat[available_cols].values
        
        mean = np.mean(feature_matrix, axis=0)
        std = np.std(feature_matrix, axis=0) + 1e-8
        normalized_matrix = (feature_matrix - mean) / std
        
        return normalized_matrix, available_cols
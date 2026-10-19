import pandas as pd
import numpy as np
from src.data.dataset import RollingWindowDatasetBuilder
from src.training.validate import chronological_train_test_split
from src.visualization.image_generator import CandlestickImageGenerator

def test_rolling_window_builder():
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.linspace(100, 150, 50),
        'high': np.linspace(102, 152, 50),
        'low': np.linspace(98, 148, 50),
        'close': np.linspace(101, 151, 50),
        'volume': np.full(50, 1000)
    })
    
    builder = RollingWindowDatasetBuilder(window_size=10, horizon=1, threshold_pct=0.5)
    X, y, returns = builder.build_dataset(df)
    
    assert len(X) == 39
    assert X.shape[1:] == (10, 5)
    assert len(y) == 39
    assert set(y).issubset({0, 1, 2})

def test_chronological_split():
    df = pd.DataFrame({'timestamp': pd.date_range('2023-01-01', periods=100)})
    train, val, test = chronological_train_test_split(df, 0.7, 0.15)
    
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert train['timestamp'].max() < val['timestamp'].min()
    assert val['timestamp'].max() < test['timestamp'].min()

def test_image_generator():
    window = np.random.rand(30, 5).astype(np.float32)
    generator = CandlestickImageGenerator(img_size=224)
    img = generator.window_to_image(window)
    
    assert img.size == (224, 224)
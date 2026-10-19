import pandas as pd
import numpy as np
from src.backtesting.backtest import BacktestEngine

def test_backtest_engine():
    df = pd.DataFrame({
        'close': [100, 102, 101, 105, 107, 106, 110]
    })
    signals = np.array([1, 1, 0, 1, 1, 0, 1])
    
    engine = BacktestEngine(initial_capital=10000.0)
    results = engine.run_backtest(df, signals)
    
    assert "final_portfolio_value" in results
    assert "sharpe_ratio" in results
    assert results["initial_capital"] == 10000.0
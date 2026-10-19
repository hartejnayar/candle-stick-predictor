import numpy as np
import pandas as pd

def calculate_trading_metrics(portfolio_values: list, initial_capital: float = 10000.0) -> dict:
    """Calculates comprehensive financial performance metrics."""
    series = pd.Series(portfolio_values)
    total_return = (series.iloc[-1] - initial_capital) / initial_capital
    daily_returns = series.pct_change().dropna()
    
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / (daily_returns.std() + 1e-8)) if len(daily_returns) > 0 else 0.0
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        "total_return_pct": float(total_return * 100),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown_pct": float(max_drawdown * 100),
        "final_capital": float(series.iloc[-1])
    }
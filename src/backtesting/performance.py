import pandas as pd
import numpy as np

def calculate_performance_summary(portfolio_values: list) -> dict:
    """Calculates drawdown, returns, and risk metrics."""
    series = pd.Series(portfolio_values)
    returns = series.pct_change().dropna()
    
    total_return = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
    annualized_return = total_return * (252 / len(series)) if len(series) > 0 else 0.0
    volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
    sharpe = (annualized_return / (volatility + 1e-8))
    
    rolling_max = series.cummax()
    max_dd = ((series - rolling_max) / rolling_max).min()
    
    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd)
    }
import pandas as pd
import numpy as np

class Backtester:
    """Quantitative backtesting engine for candlestick pattern and ML signal strategies."""
    
    def __init__(self, initial_capital: float = 10000.0, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run_backtest(self, df: pd.DataFrame, signals: np.ndarray) -> dict:
        """Simulates trading based on generated signals and calculates returns."""
        df = df.copy()
        df['signal'] = signals
        df['market_return'] = df['close'].pct_change().fillna(0)
        df['strategy_return'] = df['market_return'] * df['signal'].shift(1).fillna(0)
        
        # Apply transaction costs on signal changes
        signal_changes = df['signal'].diff().abs().fillna(0)
        df['strategy_return'] = df['strategy_return'] - (signal_changes * self.transaction_cost)
        
        df['cumulative_market'] = (1 + df['market_return']).cumprod() * self.initial_capital
        df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod() * self.initial_capital
        
        final_portfolio_value = float(df['cumulative_strategy'].iloc[-1])
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio approximation
        daily_returns = df['strategy_return']
        sharpe_ratio = float((daily_returns.mean() / (daily_returns.std() + 1e-8)) * np.sqrt(252))
        
        return {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": final_portfolio_value,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "equity_curve": df['cumulative_strategy'].tolist()
        }

# Alias for compatibility
BacktestEngine = Backtester
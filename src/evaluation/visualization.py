import matplotlib.pyplot as plt

def plot_portfolio_performance(portfolio_curve: list) -> plt.Figure:
    """Plots portfolio value growth over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(portfolio_curve, label="Portfolio Value", color="cyan", linewidth=2)
    ax.set_title("Strategy Equity Curve")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Capital ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig
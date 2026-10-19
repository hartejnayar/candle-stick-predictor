import matplotlib.pyplot as plt
import pandas as pd

def save_candlestick_plot(df: pd.DataFrame, filepath: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pd.to_datetime(df['timestamp']), df['close'], color='black', linewidth=1.5)
    ax.set_facecolor('#ffffff')
    ax.axis('off')
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
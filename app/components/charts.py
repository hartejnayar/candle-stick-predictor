import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_candlestick_chart(df: pd.DataFrame, title: str = "Price Action"):
    """Renders an interactive Plotly candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(
        x=pd.to_datetime(df['timestamp']),
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="OHLC"
    )])
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, title=title, height=400)
    st.plotly_chart(fig, use_container_width=True)
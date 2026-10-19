import streamlit as st

def render_metric_panel(label: str, value: str, delta: str = None):
    """Renders a styled metric card in Streamlit."""
    st.metric(label=label, value=value, delta=delta)
import streamlit as st

def render_explanation_panel(pattern: str, confidence: float, direction: str):
    """Renders an analytical explanation panel for model predictions and patterns."""
    st.markdown("### 🧠 AI & Pattern Analysis Breakdown")
    st.info(
        f"**Detected Pattern:** {pattern} (Confidence: {confidence*100:.1f}%)\n\n"
        f"**Core Outlook:** The multimodal fusion engine projects a **{direction}** bias based on recent price action velocity, technical indicator convergence, and historical pattern matches."
    )
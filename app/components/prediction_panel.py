import streamlit as st

def render_prediction_panel(probabilities: dict):
    """Renders probabilities breakdown for prediction classes."""
    st.markdown("### 🔮 Class Probability Distribution")
    for direction, prob in probabilities.items():
        st.progress(float(prob), text=f"{direction}: {prob*100:.1f}%")
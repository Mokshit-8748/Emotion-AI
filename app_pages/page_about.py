"""About page."""

import streamlit as st

from config import ACCENT, ACCENT2, ACCENT_GREEN


def render():
    st.markdown(f"<h1 style='color:{ACCENT};'>Project Overview</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:18px;'>"
        "A compact overview of the dashboard, the active model families, and the dataset mix behind this SER project."
        "</p>",
        unsafe_allow_html=True,
    )

    overview_cols = st.columns(3)
    overview_cards = [
        ("Primary Model", "Fine-Tuned WavLM", "best dashboard accuracy path", ACCENT),
        ("Baseline", "Original Model", "lighter handcrafted-feature comparison model", ACCENT2),
        ("Emotion Set", "7 Emotions", "angry, disgust, fear, happy, neutral, sad, surprise", ACCENT_GREEN),
    ]
    for col, (label, value, sub, color) in zip(overview_cols, overview_cards):
        with col:
            st.markdown(
                f"""
                <div class='metric-card animated-border'>
                    <h3>{label}</h3>
                    <div class='value' style='color:{color};font-size:24px;'>{value}</div>
                    <div class='sub'>{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>What This Dashboard Focuses On</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='insight-card'>
            <h3 style='color:{ACCENT_GREEN};margin-top:0;'>Current dashboard scope</h3>
            <div style='line-height:1.9;color:#c9d1d9;font-size:15px;'>
                The dashboard is centered on two practical model paths:
                the <strong style='color:{ACCENT};'>Fine-Tuned WavLM</strong> production model for
                stronger live accuracy, and the <strong>Original Model</strong> baseline for lighter comparisons.
                The pages are tuned for inference, evaluation, comparison, and training workflows rather than
                research-branch experiments that are no longer part of the main UI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Architecture Snapshot</div>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            f"""
            <div style='background:rgba(255,255,255,0.03);padding:20px;border-radius:12px;border:1px solid #30363d;'>
                <table style='width:100%;color:#c9d1d9;border-collapse:collapse;'>
                    <tr style='border-bottom:1px solid #30363d;'><td style='padding:10px;font-weight:800;color:{ACCENT};'>Layer</td><td style='padding:10px;'>Role</td></tr>
                    <tr style='border-bottom:1px dotted #21262d;'><td style='padding:8px;font-weight:700;'>Live Inference</td><td>Upload or record speech and predict emotion directly in the browser.</td></tr>
                    <tr style='border-bottom:1px dotted #21262d;'><td style='padding:8px;font-weight:700;'>Evaluation</td><td>Run fast snapshots or broader audits on detected local datasets.</td></tr>
                    <tr style='border-bottom:1px dotted #21262d;'><td style='padding:8px;font-weight:700;'>Comparison</td><td>Decide when to use WavLM versus the baseline with checkpoint and evaluation context.</td></tr>
                    <tr><td style='padding:8px;font-weight:700;'>Training</td><td>Launch either fine-tuned WavLM training or the original baseline from the dashboard.</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div style='background:#0d1117;padding:20px;border-radius:12px;border:1px solid #30363d;font-family:monospace;font-size:12px;color:#8b949e;'>
                <div style='color:#63b3ed;margin-bottom:10px;font-weight:800;'>PROJECT STRUCTURE</div>
                dashboard.py  <span style='color:#4a5568;'># entry point</span><br>
                config.py     <span style='color:#4a5568;'># shared paths and constants</span><br>
                app_pages/    <span style='color:#4a5568;'># inference, evaluation, comparison, training, about</span><br>
                src/          <span style='color:#4a5568;'># preprocessing, models, evaluation, WavLM runtime</span><br>
                models/       <span style='color:#4a5568;'># checkpoints and label assets</span><br>
                data/         <span style='color:#4a5568;'># local datasets and cached features</span><br>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Datasets Used Most Often</div>", unsafe_allow_html=True)
    dataset_cols = st.columns(3)
    dataset_cards = [
        ("RAVDESS", "Structured acted emotional speech for stable benchmarking.", ACCENT),
        ("CREMA-D", "Largest core English dataset used in the dashboard workflow.", ACCENT_GREEN),
        ("TESS", "Clean labelled speech that helps the three-dataset fine-tuning path.", ACCENT2),
    ]
    for col, (name, desc, color) in zip(dataset_cols, dataset_cards):
        with col:
            st.markdown(
                f"""
                <div class='dataset-card'>
                    <h4 style='color:{color};'>{name}</h4>
                    <div class='meta'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

"""Model comparison page."""

import json
import os

import pandas as pd
import streamlit as st

from config import ACCENT, ACCENT2, ACCENT_GREEN, ACCENT_PINK, MODEL_OPTIONS


MODEL_SPECS = {
    "Fine-Tuned WavLM": {
        "role": "Production model",
        "best_for": "Highest dashboard accuracy, difficult clips, and final evaluation passes.",
        "speed": "Heavier first load, smoother after warmup.",
        "setup": "Raw-audio runtime. No feature scaler needed for inference.",
        "tradeoff": "Slower than the baseline on long evaluation jobs.",
        "accent": ACCENT,
    },
    "Original Model": {
        "role": "Baseline model",
        "best_for": "Quick checks, lightweight demos, and legacy feature-based comparisons.",
        "speed": "Fastest dashboard loop.",
        "setup": "Uses handcrafted features and the fitted scaler.",
        "tradeoff": "Less reliable on ambiguous or noisy audio.",
        "accent": ACCENT2,
    },
}


def _runtime_label(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".pt":
        return "Fine-tuned raw-audio runtime"
    if suffix == ".h5":
        return "Classic feature runtime"
    return "Unknown"


def _size_mb(path):
    if not os.path.exists(path):
        return None
    return round(os.path.getsize(path) / (1024 * 1024), 1)


def _load_checkpoint_score(path):
    root, _ = os.path.splitext(path)
    meta_path = root + ".json"
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        score = payload.get("best_val_acc")
        return round(float(score), 2) if score is not None else None
    except Exception:
        return None


def _current_eval_snapshot():
    res = st.session_state.get("eval_results")
    if not res:
        return None
    metrics = res.get("metrics") or {}
    accuracy = metrics.get("accuracy")
    if accuracy is None:
        return None
    return {
        "model_name": res.get("model_name", "Unknown"),
        "accuracy": float(accuracy) * 100.0,
        "scope_mode": res.get("scope_mode", "Unknown"),
        "protocol_mode": res.get("protocol_mode", "Unknown"),
        "sample_count": res.get("sample_count", "Unknown"),
    }


def render(scaler, label_encoder, active_model_name):
    st.markdown(f"<h1 style='color:{ACCENT};'>Model Comparison</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#64748b;margin-bottom:20px;'>"
        "A practical side-by-side view of the models that are still active in the dashboard."
        "</div>",
        unsafe_allow_html=True,
    )

    ordered_names = [name for name in ["Fine-Tuned WavLM", "Original Model"] if name in MODEL_OPTIONS]
    if not ordered_names:
        st.warning("No dashboard models are configured for comparison.")
        return

    eval_snapshot = _current_eval_snapshot()

    top_cols = st.columns(3)
    best_model_name = "Fine-Tuned WavLM" if "Fine-Tuned WavLM" in ordered_names else ordered_names[0]
    top_cards = [
        ("Active Model", active_model_name, "current dashboard selection", ACCENT),
        (
            "Latest Dashboard Eval",
            f"{eval_snapshot['accuracy']:.2f}%" if eval_snapshot else "Not run yet",
            (
                f"{eval_snapshot['model_name']} · {eval_snapshot['scope_mode']}"
                if eval_snapshot
                else "run Model Evaluation to populate this"
            ),
            ACCENT_GREEN if eval_snapshot else ACCENT2,
        ),
        ("Recommended Default", best_model_name, "best production choice for accuracy", ACCENT),
    ]
    for col, (label, value, sub, color) in zip(top_cols, top_cards):
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

    st.markdown("<div class='section-title'>Decision Cards</div>", unsafe_allow_html=True)
    decision_cols = st.columns(len(ordered_names))
    for col, model_name in zip(decision_cols, ordered_names):
        spec = MODEL_SPECS[model_name]
        checkpoint_path = MODEL_OPTIONS[model_name]
        ready = os.path.exists(checkpoint_path)
        metadata_score = _load_checkpoint_score(checkpoint_path)
        eval_line = ""
        if eval_snapshot and eval_snapshot["model_name"] == model_name:
            eval_line = f"<br><b>Latest dashboard eval:</b> {eval_snapshot['accuracy']:.2f}%"
        elif metadata_score is not None:
            eval_line = f"<br><b>Best recorded validation:</b> {metadata_score:.2f}%"

        with col:
            st.markdown(
                f"""
                <div class='boost-card' style='height:100%;border-left-color:{spec['accent']};'>
                    <div class='rank' style='color:{spec['accent']};'>{spec['role']} · {'Ready' if ready else 'Missing'}</div>
                    <div class='title'>{model_name}</div>
                    <div class='desc'>
                        <b>Best for:</b> {spec['best_for']}<br>
                        <b>Speed:</b> {spec['speed']}<br>
                        <b>Setup:</b> {spec['setup']}<br>
                        <b>Tradeoff:</b> {spec['tradeoff']}{eval_line}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>Operational Snapshot</div>", unsafe_allow_html=True)
    rows = []
    for model_name in ordered_names:
        path = MODEL_OPTIONS[model_name]
        metadata_score = _load_checkpoint_score(path)
        rows.append(
            {
                "Model": model_name,
                "Status": "Ready" if os.path.exists(path) else "Missing",
                "Runtime": _runtime_label(path),
                "Checkpoint Size (MB)": _size_mb(path),
                "Best Recorded Validation": f"{metadata_score:.2f}%" if metadata_score is not None else "n/a",
                "Latest Dashboard Eval": (
                    f"{eval_snapshot['accuracy']:.2f}%"
                    if eval_snapshot and eval_snapshot["model_name"] == model_name
                    else "n/a"
                ),
                "Primary Use": MODEL_SPECS[model_name]["best_for"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Recommended Workflow</div>", unsafe_allow_html=True)
    workflow_cols = st.columns(3)
    workflow_cards = [
        ("Default Inference", "Fine-Tuned WavLM", "Use for live predictions when accuracy matters.", ACCENT),
        ("Quick Baseline", "Original Model", "Use when you want a lighter, faster sanity check.", ACCENT2),
        ("Best Practice", "Compare after evaluation", "Run Evaluation, then come back here to judge tradeoffs with real numbers.", ACCENT_GREEN),
    ]
    for col, (label, value, sub, color) in zip(workflow_cols, workflow_cards):
        with col:
            st.markdown(
                f"""
                <div class='metric-card animated-border'>
                    <h3>{label}</h3>
                    <div class='value' style='color:{color};font-size:23px;'>{value}</div>
                    <div class='sub'>{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>Readiness Checks</div>", unsafe_allow_html=True)
    checks = [
        ("Label Encoder", label_encoder is not None, "shared emotion label mapping", ACCENT_GREEN),
        ("Feature Scaler", scaler is not None, "required by Original Model only", ACCENT2),
    ]
    cols = st.columns(len(checks))
    for col, (label, ready, sub, color) in zip(cols, checks):
        with col:
            st.markdown(
                f"""
                <div class='metric-card animated-border'>
                    <h3>{label}</h3>
                    <div class='value' style='color:{color if ready else ACCENT_PINK};font-size:24px;'>
                        {'Ready' if ready else 'Missing'}
                    </div>
                    <div class='sub'>{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if eval_snapshot:
        st.caption(
            f"Latest evaluation snapshot: {eval_snapshot['model_name']} ran with "
            f"{eval_snapshot['protocol_mode']} / {eval_snapshot['scope_mode']} on "
            f"{eval_snapshot['sample_count']} samples."
        )
    else:
        st.caption("Run the Evaluation page to add a live benchmark snapshot to this comparison view.")

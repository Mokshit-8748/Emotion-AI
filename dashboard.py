import os
import sys
from datetime import datetime

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATASET_PATHS, EMOTIONS, EMOTION_EMOJI, ENCODER_PATH, MODEL_OPTIONS, SCALER_PATH


st.set_page_config(
    page_title="Emotion AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #080b12;
        color: #e2e8f0;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #080b12 100%);
        border-right: 1px solid #1a2233;
        min-width: 320px !important;
        max-width: 320px !important;
    }

    section[data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] .stCaption {
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        line-height: 1.3 !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f1520 0%, #141c2e 100%);
        border: 1px solid #1e2d45;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 12px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
        min-height: 115px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 14px;
        padding: 1px;
        background: linear-gradient(135deg, #63b3ed44, #b794f444, #63b3ed44);
        background-size: 200% 200%;
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: destination-out;
        mask-composite: exclude;
        animation: gradient-spin 4s ease infinite;
        pointer-events: none;
    }

    .metric-card h3 {
        margin: 0;
        font-size: 11px;
        color: #64748b;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 600;
    }

    .metric-card .value {
        font-size: 30px;
        font-weight: 800;
        color: #63b3ed;
        margin: 6px 0 2px;
        line-height: 1.1;
    }

    .metric-card .sub {
        font-size: 11px;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }

    .animated-border::before {
        animation: gradient-spin 3s ease infinite;
    }

    .boost-card {
        background: linear-gradient(135deg, #0a1018 0%, #111928 100%);
        border: 1px solid #1e2d45;
        border-left: 4px solid #63b3ed;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }

    .boost-card .rank {
        font-size: 10px;
        color: #64748b;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .boost-card .title {
        font-size: 17px;
        font-weight: 800;
        color: #f1f5f9;
        margin: 4px 0 6px;
    }

    .boost-card .desc {
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.65;
    }

    .dataset-card {
        background: linear-gradient(135deg, #091510 0%, #0c1e16 100%);
        border: 1px solid #163024;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }

    .dataset-card h4 {
        margin: 0 0 6px;
        color: #68d391;
        font-size: 15px;
    }

    .dataset-card .meta {
        font-size: 12px;
        color: #94a3b8;
        line-height: 1.5;
    }

    .section-title {
        font-size: 28px;
        font-weight: 800;
        color: #f0f6fc;
        margin: 36px 0 18px 0;
        letter-spacing: -0.5px;
        border-left: 5px solid #63b3ed;
        padding-left: 15px;
    }

    .insight-card {
        background: rgba(13, 17, 23, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }

    .diversity-bar-bg {
        background: #131c2e;
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin: 3px 0 8px;
    }

    .diversity-bar-fill {
        height: 8px;
        border-radius: 6px;
        transition: width 0.5s ease;
    }

    .stAlert, .stCode, .stCodeBlock {
        border-radius: 10px !important;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    @keyframes gradient-spin {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _is_dashboard_compatible_model(path):
    base = os.path.basename(path).lower()
    return (
        (path.lower().endswith(".h5") and base.startswith("emotion_ser_model"))
        or (path.lower().endswith(".pt") and "emotion_wavlm_finetuned" in base)
    )


def _download_if_missing(path):
    repo_id = os.environ.get("HF_MODEL_REPO")
    if repo_id and not os.path.exists(path):
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=repo_id,
                filename=os.path.basename(path),
                local_dir=os.path.dirname(path)
            )
            return True
        except Exception as e:
            print(f"HF Download failed for {path}: {e}")
            return False
    return os.path.exists(path)


DASHBOARD_MODEL_OPTIONS = {
    name: path
    for name, path in MODEL_OPTIONS.items()
    if _is_dashboard_compatible_model(path) and (os.path.exists(path) or os.environ.get("HF_MODEL_REPO"))
}


@st.cache_resource(show_spinner="Loading model...")
def load_ser_model(path):
    try:
        if not os.path.exists(path):
            if not _download_if_missing(path):
                return None, f"Model not found at `{path}`"

        if path.lower().endswith(".pt"):
            from src.wavlm_runtime import load_wavlm_runtime

            return load_wavlm_runtime(path), None

        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            from tf_keras.models import load_model

        try:
            from src.model import ResidualBlock, SEBlock, TemporalAttention

            custom_objects = {
                "TemporalAttention": TemporalAttention,
                "SEBlock": SEBlock,
                "ResidualBlock": ResidualBlock,
            }
        except Exception:
            custom_objects = {}

        if not os.path.exists(path):
            if not _download_if_missing(path):
                return None, f"Model not found at `{path}`"

        model = load_model(path, custom_objects=custom_objects, compile=False)
        setattr(model, "ser_backend", "keras_feature")
        return model, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Loading label encoder...")
def load_label_encoder():
    try:
        import joblib

        if not os.path.exists(ENCODER_PATH):
            if not _download_if_missing(ENCODER_PATH):
                return None, f"Not found: `{ENCODER_PATH}`"
        return joblib.load(ENCODER_PATH), None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Loading scaler...")
def load_scaler():
    try:
        import joblib

        if not os.path.exists(SCALER_PATH):
            if not _download_if_missing(SCALER_PATH):
                return None
        return joblib.load(SCALER_PATH)
    except Exception:
        return None


label_encoder, encoder_err = load_label_encoder()
scaler = load_scaler()


def _get_system_stats():
    stats = {"cpu": None, "ram": None, "ram_gb": None, "gpu_count": 0}
    try:
        import psutil

        stats["cpu"] = psutil.cpu_percent(interval=0.2)
        stats["ram"] = psutil.virtual_memory().percent
        stats["ram_gb"] = psutil.virtual_memory().used / (1024 ** 3)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        stats["gpu_count"] = len(gpus)
    except Exception:
        pass

    return stats


def _model_file_info(path):
    if not os.path.exists(path):
        return None, None
    size_mb = os.path.getsize(path) / (1024 * 1024)
    modified = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    return size_mb, modified


def _status_row(label, ok, detail=""):
    dot = "●" if ok else "○"
    color = "#68d391" if ok else "#fc8181"
    suffix = (
        f" <span style='color:#64748b;font-size:10px;font-family:monospace;'>{detail}</span>"
        if detail
        else ""
    )
    st.markdown(
        f"<span style='color:{color};font-size:12px;'>{dot} {label}</span>{suffix}",
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;padding:12px 0 6px;'>
          <div style='font-size:36px;'>🎭</div>
          <div style='font-size:20px;font-weight:800;color:#f1f5f9;letter-spacing:1px;'>Emotion AI</div>
          <div style='font-size:10px;color:#64748b;letter-spacing:3px;text-transform:uppercase;margin-top:2px;'>
            Production Dashboard
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title' style='margin-top:14px;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "",
        ["Inference", "Evaluation", "Comparison", "Training", "About"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='section-title'>Model Selection</div>", unsafe_allow_html=True)
    if not DASHBOARD_MODEL_OPTIONS:
        st.error("No compatible dashboard models were found in the models folder.")
        st.stop()

    available_model_names = list(DASHBOARD_MODEL_OPTIONS.keys())
    if st.session_state.get("model_choice") not in available_model_names:
        st.session_state["model_choice"] = available_model_names[0]

    model_choice = st.selectbox(
        "Select SER model",
        options=available_model_names,
        index=available_model_names.index(st.session_state["model_choice"]),
        key="model_choice",
        label_visibility="collapsed",
    )
    model_path = DASHBOARD_MODEL_OPTIONS[model_choice]

    accuracy_hint_model = "Fine-Tuned WavLM"
    if model_choice == "Original Model" and accuracy_hint_model in DASHBOARD_MODEL_OPTIONS:
        if st.session_state.get("accuracy_hint_seen_for") != model_choice:
            if hasattr(st, "toast"):
                st.toast("For best accuracy, switch to Fine-Tuned WavLM.", icon="🎯")
            st.session_state["accuracy_hint_seen_for"] = model_choice
        st.info("For best accuracy, use Fine-Tuned WavLM when response quality matters more than raw speed.")
    elif st.session_state.get("accuracy_hint_seen_for") != model_choice:
        st.session_state["accuracy_hint_seen_for"] = model_choice

    dataset_signature = ",".join(DATASET_PATHS)
    if st.session_state.get("last_model_choice") != model_choice:
        st.session_state["last_model_choice"] = model_choice
        st.session_state["eval_results"] = None
    if st.session_state.get("last_dataset_paths") != dataset_signature:
        st.session_state["last_dataset_paths"] = dataset_signature
        st.session_state["eval_results"] = None

    model_exists = os.path.exists(model_path)
    st.markdown(
        f"<span style='color:{'#68d391' if model_exists else '#fc8181'};font-size:11px;font-family:monospace;'>"
        f"{'✅' if model_exists else '❌'} {'checkpoint ready' if model_exists else 'checkpoint missing'}</span>",
        unsafe_allow_html=True,
    )

    model, model_err = load_ser_model(model_path)

    st.markdown("---")
    st.markdown("<div class='section-title'>System Status</div>", unsafe_allow_html=True)
    _status_row(f"SER Model ({model_choice})", model is not None, model_err or "")
    _status_row("Label Encoder", label_encoder is not None, encoder_err or "")
    _status_row("Scaler", scaler is not None, "used by Original Model only" if scaler is not None else "optional for WavLM")

    stats = _get_system_stats()
    if stats["cpu"] is not None:
        st.markdown(
            f"<span style='font-size:11px;color:#64748b;'>CPU</span> "
            f"<span style='font-size:11px;font-family:monospace;color:#63b3ed;'>{stats['cpu']:.0f}%</span>"
            f"<span style='font-size:11px;color:#64748b;'> | RAM</span> "
            f"<span style='font-size:11px;font-family:monospace;color:#63b3ed;'>{stats['ram']:.0f}% ({stats['ram_gb']:.1f}GB)</span>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<span style='font-size:11px;color:{'#68d391' if stats['gpu_count'] else '#64748b'};'>"
        f"GPU: {stats['gpu_count']} device(s)</span>",
        unsafe_allow_html=True,
    )

    size_mb, modified = _model_file_info(model_path)
    if size_mb is not None:
        st.markdown(
            f"<span style='font-size:10px;color:#64748b;font-family:monospace;'>"
            f"Size: {size_mb:.1f} MB | Modified: {modified}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("<div class='section-title'>Supported Emotions</div>", unsafe_allow_html=True)
    for emotion in EMOTIONS:
        st.markdown(f"{EMOTION_EMOJI[emotion]} **{emotion.capitalize()}**")


if page == "Inference":
    from app_pages.page_inference import render as render_inference

    render_inference(model, label_encoder, scaler, model_choice, model_err)
elif page == "Evaluation":
    from app_pages.page_evaluation import render as render_evaluation

    render_evaluation(model, label_encoder, scaler, model_choice, model_path, model_err)
elif page == "Comparison":
    from app_pages.page_comparison import render as render_comparison

    render_comparison(scaler, label_encoder, model_choice)
elif page == "Training":
    from app_pages.page_training import render as render_training

    render_training()
elif page == "About":
    from app_pages.page_about import render as render_about

    render_about()

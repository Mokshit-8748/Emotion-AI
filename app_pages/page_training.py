"""Interactive training page for the dashboard."""

import json
import os
import re
import subprocess
import sys
import time

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DATASET_PATHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    MODEL_PATH_BASE,
    MODEL_PATH_FINETUNE_DASHBOARD,
)


WAVLM_OUTPUT_DEFAULT = os.path.join(PROJECT_ROOT, "models", "emotion_wavlm_finetuned_dashboard_run.pt")
BASELINE_OUTPUT_DEFAULT = MODEL_PATH_BASE
WAVLM_SUMMARY_RE = re.compile(
    r"EPOCH\s+(?P<epoch>\d+):\s+train_loss=(?P<loss>[0-9.]+)\s+"
    r"train_acc=(?P<accuracy>[0-9.]+)%\s+val_loss=(?P<val_loss>[0-9.]+)\s+"
    r"val_acc=(?P<val_accuracy>[0-9.]+)%"
)
WAVLM_DONE_RE = re.compile(r"Training complete\. Best validation accuracy:\s*(?P<best>[0-9.]+)%")
BASELINE_DONE_RE = re.compile(r'"best_val_accuracy":\s*(?P<best>[0-9.]+)')


def _blank_history():
    return {"epoch": [], "loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}


def _ensure_state():
    defaults = {
        "training_proc": None,
        "training_history": _blank_history(),
        "training_log": [],
        "training_done": False,
        "training_target_epochs": 0,
        "training_mode_name": "",
        "training_summary": {},
        "training_output": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _dataset_label_map(paths):
    mapping = {}
    for path in paths:
        name = os.path.basename(os.path.normpath(path)) or path
        if name in mapping:
            name = path
        mapping[name] = path
    return mapping


def _default_dataset_selection():
    preferred = {"RAVDESS", "CREMA-D", "TESS"}
    selected = [path for path in DATASET_PATHS if os.path.basename(os.path.normpath(path)) in preferred]
    return selected or list(DATASET_PATHS)


def _combine_dataset_paths(selected_detected, custom_paths_raw):
    active = list(selected_detected)
    for line in custom_paths_raw.splitlines():
        candidate = line.strip()
        if candidate and os.path.isdir(candidate) and candidate not in active:
            active.append(candidate)
    return active


def _metadata_path_for(output_path):
    root, _ = os.path.splitext(output_path)
    return root + ".json"


def _build_wavlm_command(config):
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "train_finetune_3dataset.py"),
        "--device",
        config["device"],
        "--datasets",
        ",".join(config["datasets"]),
        "--init-from",
        config["init_from"],
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--grad-accum-steps",
        str(config["grad_accum_steps"]),
        "--lr-backbone",
        str(config["lr_backbone"]),
        "--lr-layer-weights",
        str(config["lr_layer_weights"]),
        "--lr-head",
        str(config["lr_head"]),
        "--label-smoothing",
        str(config["label_smoothing"]),
        "--mixup-prob",
        str(config["mixup_prob"]),
        "--unfreeze-layers",
        str(config["unfreeze_layers"]),
        "--val-split",
        str(config["val_split"]),
        "--patience",
        str(config["patience"]),
        "--output",
        config["output"],
        "--metadata",
        config["metadata"],
    ]
    if not config["balance_sampler"]:
        cmd.append("--no-balance-sampler")
    return cmd


def _build_baseline_command(config):
    return [
        sys.executable,
        os.path.join(PROJECT_ROOT, "src", "train_model.py"),
        "--datasets",
        ",".join(config["datasets"]),
        "--epochs",
        str(config["epochs"]),
        "--batch_size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--dropout",
        str(config["dropout"]),
        "--output",
        config["output"],
    ]


def _append_baseline_epoch(history, payload):
    history["epoch"].append(int(payload["epoch"]))
    history["loss"].append(float(payload["loss"]))
    history["val_loss"].append(float(payload["val_loss"]))
    history["accuracy"].append(float(payload["accuracy"]))
    history["val_accuracy"].append(float(payload["val_accuracy"]))


def _append_wavlm_epoch(history, match):
    history["epoch"].append(int(match.group("epoch")))
    history["loss"].append(float(match.group("loss")))
    history["val_loss"].append(float(match.group("val_loss")))
    history["accuracy"].append(float(match.group("accuracy")) / 100.0)
    history["val_accuracy"].append(float(match.group("val_accuracy")) / 100.0)


def _poll_training_output(training_family):
    proc = st.session_state["training_proc"]
    history = st.session_state["training_history"]
    deadline = time.time() + 2.0

    while time.time() < deadline:
        line = proc.stdout.readline() if proc and proc.stdout else ""
        if not line:
            if proc and proc.poll() is not None:
                st.session_state["training_done"] = True
                st.session_state["training_proc"] = None
                break
            time.sleep(0.05)
            continue

        line = line.rstrip()
        st.session_state["training_log"].append(line)

        if training_family == "baseline" and line.startswith("[EPOCH]"):
            try:
                payload = json.loads(line[len("[EPOCH]") :].strip())
                _append_baseline_epoch(history, payload)
            except Exception:
                pass
        else:
            match = WAVLM_SUMMARY_RE.search(line)
            if match:
                _append_wavlm_epoch(history, match)

        if line.startswith("[DONE]"):
            match = BASELINE_DONE_RE.search(line)
            if match:
                st.session_state["training_summary"]["best_val_accuracy"] = float(match.group("best"))
            st.session_state["training_done"] = True
            st.session_state["training_proc"] = None

        wavlm_done = WAVLM_DONE_RE.search(line)
        if wavlm_done:
            st.session_state["training_summary"]["best_val_accuracy"] = float(wavlm_done.group("best"))
            st.session_state["training_done"] = True
            st.session_state["training_proc"] = None

    if proc and proc.poll() is not None:
        st.session_state["training_done"] = True
        st.session_state["training_proc"] = None


def _render_live_metrics():
    history = st.session_state["training_history"]
    if not history["epoch"]:
        return

    import pandas as pd

    latest_epoch = history["epoch"][-1]
    target_epochs = max(1, int(st.session_state["training_target_epochs"] or latest_epoch))
    latest_val_acc = history["val_accuracy"][-1] * 100.0
    latest_val_loss = history["val_loss"][-1]
    st.progress(
        min(1.0, latest_epoch / target_epochs),
        text=f"Epoch {latest_epoch}/{target_epochs} - val_acc {latest_val_acc:.2f}% - val_loss {latest_val_loss:.4f}",
    )

    df_acc = pd.DataFrame(
        {"Train Accuracy": history["accuracy"], "Val Accuracy": history["val_accuracy"]},
        index=history["epoch"],
    )
    df_loss = pd.DataFrame(
        {"Train Loss": history["loss"], "Val Loss": history["val_loss"]},
        index=history["epoch"],
    )

    st.markdown("**Accuracy**")
    st.line_chart(df_acc, use_container_width=True)
    st.markdown("**Loss**")
    st.line_chart(df_loss, use_container_width=True)


def _render_training_results():
    history = st.session_state["training_history"]
    if not history["epoch"]:
        return

    best_idx = max(range(len(history["val_accuracy"])), key=lambda idx: history["val_accuracy"][idx])
    best_val_acc = history["val_accuracy"][best_idx] * 100.0
    best_epoch = history["epoch"][best_idx]
    last_epoch = history["epoch"][-1]

    col1, col2, col3 = st.columns(3)
    cards = [
        ("Best Val Accuracy", f"{best_val_acc:.2f}%", f"epoch {best_epoch}", "#68d391"),
        ("Epochs Completed", str(last_epoch), st.session_state["training_mode_name"], "#63b3ed"),
        ("Output", os.path.basename(st.session_state["training_output"]) or "n/a", "latest checkpoint target", "#f6ad55"),
    ]
    for col, (label, value, sub, color) in zip([col1, col2, col3], cards):
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

    _render_live_metrics()


def render():
    _ensure_state()

    st.markdown("# 🏋️ Training Control Center")
    st.markdown(
        "Train the **Fine-Tuned WavLM** production model or the **Original Model** baseline "
        "directly from the dashboard. WavLM is the recommended path for the best accuracy."
    )
    st.markdown("---")

    training_mode = st.radio(
        "Training Target",
        ["Fine-Tuned WavLM", "Original Model"],
        horizontal=True,
    )
    training_family = "wavlm" if training_mode == "Fine-Tuned WavLM" else "baseline"

    detected_map = _dataset_label_map(DATASET_PATHS)
    default_paths = _default_dataset_selection()
    default_labels = [label for label, path in detected_map.items() if path in default_paths]

    st.markdown("### Dataset Sources")
    selected_detected = st.multiselect(
        "Detected dataset folders",
        options=list(detected_map.keys()),
        default=default_labels,
        help="The defaults match the three-dataset fine-tuning setup that performed best in this project.",
    )
    custom_paths_raw = st.text_area(
        "Optional extra local dataset paths (one per line)",
        value="",
        help="Use this on another machine if some datasets live outside the project data folder.",
    )
    active_paths = _combine_dataset_paths([detected_map[label] for label in selected_detected], custom_paths_raw)

    if active_paths:
        st.caption(f"{len(active_paths)} dataset folder(s) selected.")
    else:
        st.warning("Select at least one valid dataset folder before starting training.")

    st.markdown("### Training Configuration")

    if training_family == "wavlm":
        st.info("Fine-Tuned WavLM is the recommended production training path. The default output is a safe new checkpoint, so your current best model stays untouched.")
        col1, col2 = st.columns(2)
        with col1:
            device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
            epochs = st.slider("Epochs", min_value=5, max_value=60, value=25, step=5)
            batch_size = st.select_slider("Batch Size", options=[2, 4, 8, 16], value=8)
            grad_accum_steps = st.select_slider("Gradient Accumulation", options=[1, 2, 4, 8, 16], value=8)
            unfreeze_layers = st.slider("Unfreeze Layers", min_value=4, max_value=12, value=12, step=1)
        with col2:
            lr_backbone = st.select_slider("Backbone LR", options=[1e-6, 2e-6, 3e-6, 5e-6], value=2e-6, format_func=lambda x: f"{x:.0e}")
            lr_layer_weights = st.select_slider("Layer-Weight LR", options=[1e-6, 2e-6, 3e-6, 5e-6], value=2e-6, format_func=lambda x: f"{x:.0e}")
            lr_head = st.select_slider("Head LR", options=[5e-5, 8e-5, 1e-4, 1.5e-4, 2e-4], value=1e-4, format_func=lambda x: f"{x:.1e}")
            mixup_prob = st.slider("Mixup Probability", min_value=0.0, max_value=0.45, value=0.10, step=0.05)
            label_smoothing = st.slider("Label Smoothing", min_value=0.0, max_value=0.05, value=0.01, step=0.01)

        col3, col4 = st.columns(2)
        with col3:
            val_split = st.slider("Validation Split", min_value=0.10, max_value=0.20, value=0.10, step=0.01)
            patience = st.slider("Early-Stop Patience", min_value=4, max_value=12, value=10, step=1)
            balance_sampler = st.checkbox("Use balanced sampler", value=True)
        with col4:
            init_from = st.text_input("Warm-start checkpoint", value=MODEL_PATH_FINETUNE_DASHBOARD)
            output = st.text_input("Output checkpoint", value=WAVLM_OUTPUT_DEFAULT)
            metadata = st.text_input("Metadata output", value=_metadata_path_for(output))

        config = {
            "device": device,
            "datasets": active_paths,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "lr_backbone": lr_backbone,
            "lr_layer_weights": lr_layer_weights,
            "lr_head": lr_head,
            "mixup_prob": mixup_prob,
            "label_smoothing": label_smoothing,
            "unfreeze_layers": unfreeze_layers,
            "val_split": val_split,
            "patience": patience,
            "balance_sampler": balance_sampler,
            "init_from": init_from,
            "output": output,
            "metadata": metadata,
        }
        command = _build_wavlm_command(config)
    else:
        st.info("Original Model is the lighter baseline. Keep it for comparisons and legacy feature-based checks, not for the best final accuracy.")
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", min_value=20, max_value=200, value=min(DEFAULT_EPOCHS, 120), step=10)
            batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=DEFAULT_BATCH_SIZE)
        with col2:
            lr = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
                value=DEFAULT_LR if DEFAULT_LR in [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3] else 3e-4,
                format_func=lambda x: f"{x:.0e}",
            )
            dropout = st.slider("Dropout", min_value=0.10, max_value=0.60, value=DEFAULT_DROPOUT, step=0.05)

        output = st.text_input("Output checkpoint", value=BASELINE_OUTPUT_DEFAULT)
        config = {
            "datasets": active_paths,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
            "output": output,
        }
        command = _build_baseline_command(config)

    st.markdown("### Run Summary")
    summary_cols = st.columns(3)
    summary_cards = [
        ("Training Target", training_mode, "dashboard-managed run", "#63b3ed"),
        ("Datasets", str(len(active_paths)), "selected folder(s)", "#68d391"),
        ("Output", os.path.basename(config["output"]) or "n/a", "checkpoint target", "#f6ad55"),
    ]
    for col, (label, value, sub, color) in zip(summary_cols, summary_cards):
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

    if training_family == "wavlm" and os.path.normpath(config["output"]) == os.path.normpath(MODEL_PATH_FINETUNE_DASHBOARD):
        st.warning("This run will overwrite the checkpoint currently used by the dashboard.")
    if training_family == "baseline" and os.path.normpath(config["output"]) == os.path.normpath(MODEL_PATH_BASE):
        st.warning("This run will overwrite the baseline checkpoint currently used by the dashboard.")

    with st.expander("Command Preview"):
        st.code(" ".join(command), language="powershell")

    start_col, stop_col = st.columns([1, 1])
    with start_col:
        start = st.button("Start Training", type="primary", use_container_width=True, disabled=not active_paths)
    with stop_col:
        stop = st.button("Stop Current Run", use_container_width=True, disabled=st.session_state["training_proc"] is None)

    if stop and st.session_state["training_proc"] is not None:
        try:
            st.session_state["training_proc"].terminate()
            st.session_state["training_proc"] = None
            st.session_state["training_done"] = True
            st.warning("Training stopped by user.")
        except Exception as exc:
            st.error(f"Could not stop the training process: {exc}")

    if start:
        st.session_state["training_history"] = _blank_history()
        st.session_state["training_log"] = []
        st.session_state["training_done"] = False
        st.session_state["training_target_epochs"] = config["epochs"]
        st.session_state["training_mode_name"] = training_mode
        st.session_state["training_summary"] = {}
        st.session_state["training_output"] = config["output"]

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=PROJECT_ROOT,
            )
            st.session_state["training_proc"] = proc
            st.success(f"{training_mode} run started.")
        except Exception as exc:
            st.error(f"Failed to start training: {exc}")
            return

    if st.session_state["training_proc"] is not None and not st.session_state["training_done"]:
        st.markdown("---")
        st.markdown("### Live Training Output")
        _poll_training_output(training_family)
        _render_live_metrics()
        st.code("\n".join(st.session_state["training_log"][-60:]), language="text")

        if st.session_state["training_done"]:
            st.success("Training complete. Latest best checkpoint is ready.")
        else:
            st.info("Training is still running. This page will refresh automatically.")
            time.sleep(1)
            st.rerun()

    elif st.session_state["training_done"]:
        st.markdown("---")
        st.success("Training complete.")
        _render_training_results()
        st.code("\n".join(st.session_state["training_log"][-60:]), language="text")

        if st.button("Start Fresh Run", use_container_width=True):
            st.session_state["training_history"] = _blank_history()
            st.session_state["training_log"] = []
            st.session_state["training_done"] = False
            st.session_state["training_target_epochs"] = 0
            st.session_state["training_mode_name"] = ""
            st.session_state["training_summary"] = {}
            st.session_state["training_output"] = ""
            st.rerun()

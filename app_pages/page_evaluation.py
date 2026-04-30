"""Model Evaluation page."""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from config import ACCENT, ACCENT2, ACCENT_GREEN, ACCENT_PINK, DATASET_PATHS, DEFAULT_DATASET_NAMES, EMOTION_COLORS, EMOTION_EMOJI
from src.dataset_index import collect_file_pairs
from src.dataset_builder import build_multi_dataset
from src.evaluate_model import (
    compute_roc_curves,
    compute_roc_curves_from_probabilities,
    evaluate_model,
    generate_pdf_report,
    summarize_predictions,
)


@st.cache_data(show_spinner=False)
def _load_feature_dataset_cached(active_paths):
    return build_multi_dataset(list(active_paths))


@st.cache_data(show_spinner=False)
def _collect_audio_pairs_cached(active_paths):
    file_pairs = []
    for path in active_paths:
        file_pairs.extend(collect_file_pairs(path))
    return file_pairs


def _sample_feature_dataset(X, y, per_class_limit, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    selected = []
    for emotion in np.unique(y):
        indices = np.flatnonzero(y == emotion)
        if len(indices) > per_class_limit:
            indices = rng.choice(indices, size=per_class_limit, replace=False)
        selected.extend(indices.tolist())
    selected = np.array(sorted(selected), dtype=np.int64)
    return X[selected], y[selected]


def _sample_file_pairs(file_pairs, per_class_limit, seed=42):
    rng = np.random.default_rng(seed)
    grouped = {}
    for file_path, emotion in file_pairs:
        grouped.setdefault(emotion, []).append((file_path, emotion))

    sampled = []
    for emotion in sorted(grouped):
        items = grouped[emotion]
        if len(items) > per_class_limit:
            choice = rng.choice(len(items), size=per_class_limit, replace=False)
            sampled.extend([items[i] for i in sorted(choice)])
        else:
            sampled.extend(items)
    return sampled


def _dataset_label_map(paths):
    mapping = {}
    for path in paths:
        label = os.path.basename(os.path.normpath(path)) or path
        if label in mapping:
            label = path
        mapping[label] = path
    return mapping


def _infer_dataset_name(file_path):
    normalized_parts = [part.upper() for part in os.path.normpath(file_path).split(os.sep)]
    for dataset_name in DEFAULT_DATASET_NAMES:
        if dataset_name.upper() in normalized_parts:
            return dataset_name
    return os.path.basename(os.path.dirname(file_path)) or "Unknown"


def _infer_group_key(file_path):
    dataset_name = _infer_dataset_name(file_path)
    file_name = os.path.basename(file_path)
    stem = os.path.splitext(file_name)[0]
    parent_dir = os.path.basename(os.path.dirname(file_path))

    if dataset_name == "RAVDESS":
        return f"{dataset_name}:actor_{stem.split('-')[-1]}"
    if dataset_name == "CREMA-D":
        return f"{dataset_name}:speaker_{stem.split('_')[0]}"
    if dataset_name == "TESS":
        parent_upper = parent_dir.upper()
        if parent_upper.startswith("OAF"):
            return f"{dataset_name}:speaker_OAF"
        if parent_upper.startswith("YAF"):
            return f"{dataset_name}:speaker_YAF"
        return f"{dataset_name}:speaker_{parent_dir.split('_')[0]}"
    if dataset_name == "EMO-DB":
        return f"{dataset_name}:speaker_{stem[:2]}"
    if dataset_name == "IEMOCAP":
        for part in os.path.normpath(file_path).split(os.sep):
            if part.lower().startswith("session"):
                return f"{dataset_name}:{part}"
        return f"{dataset_name}:{stem[:6]}"

    return f"{dataset_name}:{parent_dir or 'group'}"


def _apply_group_audit(file_pairs, holdout_ratio=0.25, seed=42):
    rng = np.random.default_rng(seed)
    dataset_groups = {}
    for file_path, emotion in file_pairs:
        dataset_name = _infer_dataset_name(file_path)
        group_key = _infer_group_key(file_path)
        dataset_groups.setdefault(dataset_name, {}).setdefault(group_key, []).append((file_path, emotion))

    selected_pairs = []
    summary_rows = []
    for dataset_name in sorted(dataset_groups):
        group_map = dataset_groups[dataset_name]
        group_names = sorted(group_map)
        if len(group_names) <= 1:
            chosen_groups = group_names
        else:
            holdout_count = max(1, int(np.ceil(len(group_names) * holdout_ratio)))
            chosen_idx = rng.choice(len(group_names), size=holdout_count, replace=False)
            chosen_groups = [group_names[i] for i in sorted(chosen_idx)]

        sample_count = 0
        for group_name in chosen_groups:
            group_items = group_map[group_name]
            sample_count += len(group_items)
            selected_pairs.extend(group_items)

        summary_rows.append(
            {
                "Dataset": dataset_name,
                "Groups Used": len(chosen_groups),
                "Total Groups": len(group_names),
                "Samples": sample_count,
            }
        )

    return selected_pairs, summary_rows


def _estimate_scope_samples(active_paths, scope_mode, per_class_limit, protocol_mode="Direct Folder Snapshot", holdout_ratio=0.25):
    if not active_paths:
        return 0
    file_pairs = _collect_audio_pairs_cached(tuple(active_paths))
    if protocol_mode == "Speaker/Session Audit":
        file_pairs, _ = _apply_group_audit(file_pairs, holdout_ratio=holdout_ratio)
    if scope_mode == "Full Dataset":
        return len(file_pairs)
    counts = Counter(emotion for _, emotion in file_pairs)
    return sum(min(count, per_class_limit) for count in counts.values())


def _confidence_summary(probabilities, predictions, y_true):
    probs = np.asarray(probabilities)
    preds = np.asarray(predictions)
    truth = np.asarray(y_true)

    top1 = np.max(probs, axis=1)
    if probs.shape[1] > 1:
        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        margins = top1

    correct_mask = preds == truth
    low_conf_mask = top1 < 0.60

    def _safe_mean(values, mask=None):
        if mask is not None:
            values = values[mask]
        return float(np.mean(values)) if len(values) else 0.0

    return {
        "avg_confidence": _safe_mean(top1) * 100.0,
        "avg_margin": _safe_mean(margins) * 100.0,
        "correct_confidence": _safe_mean(top1, correct_mask) * 100.0,
        "incorrect_confidence": _safe_mean(top1, ~correct_mask) * 100.0,
        "low_conf_share": float(np.mean(low_conf_mask)) * 100.0 if len(low_conf_mask) else 0.0,
    }


def _top_confusion_pairs(confusion_matrix, classes, limit=5):
    rows = []
    for actual_idx, actual_name in enumerate(classes):
        for predicted_idx, predicted_name in enumerate(classes):
            if actual_idx == predicted_idx:
                continue
            count = int(confusion_matrix[actual_idx][predicted_idx])
            if count <= 0:
                continue
            rows.append({"Actual": actual_name, "Predicted": predicted_name, "Count": count})
    rows.sort(key=lambda row: row["Count"], reverse=True)
    return rows[:limit]


def render(model, label_encoder, scaler, model_name, model_path, model_err=None):
    st.markdown(f"<h1 style='color:{ACCENT};'>Model Evaluation</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:#64748b;margin-bottom:20px;'>Responsive evaluation tools for <b>{model_name}</b></div>",
        unsafe_allow_html=True,
    )

    if model is None:
        if os.path.exists(model_path):
            st.error(f"Unable to load model from `{model_path}`.")
            if model_err:
                st.code(model_err)
        else:
            st.error(f"Model file not found at `{model_path}`. Please train the model first.")
        return

    if "eval_results" not in st.session_state:
        st.session_state["eval_results"] = None

    backend = getattr(model, "ser_backend", "keras_feature")
    detected_dataset_map = _dataset_label_map(DATASET_PATHS)

    with st.expander("Dataset Configuration", expanded=st.session_state["eval_results"] is None):
        path_mode = st.radio("Dataset Path Mode", ["Detected Project Datasets", "Custom Paths"], index=0, horizontal=True)
        protocol_options = ["Direct Folder Snapshot"]
        if backend == "wavlm_pt":
            protocol_options.append("Speaker/Session Audit")
        protocol_mode = st.radio(
            "Evaluation Protocol",
            protocol_options,
            index=0,
            horizontal=True,
            help="Speaker/Session Audit is available for the Fine-Tuned WavLM runtime and evaluates inferred speaker/session groups from the current folders.",
        )
        holdout_ratio = st.slider(
            "Audit hold-out share",
            min_value=0.10,
            max_value=0.50,
            value=0.25,
            step=0.05,
            disabled=protocol_mode != "Speaker/Session Audit",
        )
        scope_mode = st.radio(
            "Evaluation Scope",
            ["Fast Snapshot", "Full Dataset"],
            index=0,
            horizontal=True,
            help="Fast Snapshot is recommended for quick feedback. Full Dataset is slower, especially for the Fine-Tuned WavLM model.",
        )
        per_class_limit = st.slider(
            "Fast Snapshot: max samples per emotion",
            min_value=20,
            max_value=200,
            value=80,
            step=20,
            disabled=scope_mode != "Fast Snapshot",
        )

        if path_mode == "Detected Project Datasets":
            if not detected_dataset_map:
                st.warning("No project-relative datasets were detected in the current workspace. Use Custom Paths below.")
                active_paths = []
            else:
                selected_datasets = st.multiselect(
                    "Detected dataset folders",
                    options=list(detected_dataset_map.keys()),
                    default=list(detected_dataset_map.keys()),
                )
                active_paths = [detected_dataset_map[name] for name in selected_datasets]
                st.info(
                    "Using detected dataset folders from the current workspace. "
                    "Switch to Custom Paths if another machine stores the datasets elsewhere."
                )
        else:
            custom_paths_raw = st.text_area(
                "Enter Local Paths (one per line)",
                value="\n".join(DATASET_PATHS) if DATASET_PATHS else "",
                help="Paste absolute paths to your dataset folders on the current machine.",
            )
            active_paths = [p.strip() for p in custom_paths_raw.split("\n") if p.strip()]
            valid_paths = [p for p in active_paths if os.path.isdir(p)]
            if not valid_paths:
                st.error("No valid directory paths provided.")
                active_paths = []
            elif len(valid_paths) < len(active_paths):
                st.warning(f"{len(active_paths) - len(valid_paths)} path(s) are invalid or missing.")
                active_paths = valid_paths
            else:
                st.success(f"Locked in {len(valid_paths)} valid dataset folder(s).")
                active_paths = valid_paths

        if scope_mode == "Fast Snapshot":
            st.caption("Fast Snapshot samples a balanced subset from each emotion to keep the page responsive.")
        else:
            st.caption("Full Dataset uses every available sample and can take a while on larger folders.")

        if protocol_mode == "Speaker/Session Audit":
            st.info(
                "Speaker/Session Audit evaluates inferred speaker or session groups from the current folders. "
                "It is better for spotting generalization issues, but it is not guaranteed to be fully unseen "
                "relative to the checkpoint's original training split."
            )

        estimated_samples = (
            _estimate_scope_samples(active_paths, scope_mode, per_class_limit, protocol_mode, holdout_ratio)
            if active_paths
            else 0
        )
        if estimated_samples:
            workload_note = "heavier raw-audio pass" if backend == "wavlm_pt" else "feature-based pass"
            st.caption(f"Estimated evaluation size: ~{estimated_samples} samples ({workload_note}).")

        if st.button("Run Evaluation", type="primary", use_container_width=True, disabled=not active_paths):
            with st.spinner("Running evaluation..."):
                try:
                    if backend == "wavlm_pt":
                        file_pairs = _collect_audio_pairs_cached(tuple(active_paths))
                        if not file_pairs:
                            raise RuntimeError("No usable audio files were found in the selected dataset folders.")
                        audit_summary = None
                        if protocol_mode == "Speaker/Session Audit":
                            file_pairs, audit_summary = _apply_group_audit(file_pairs, holdout_ratio=holdout_ratio)
                        if scope_mode == "Fast Snapshot":
                            file_pairs = _sample_file_pairs(file_pairs, per_class_limit)
                        y_test_encoded, probabilities = model.predict_file_pairs(file_pairs, batch_size=8)
                        if len(y_test_encoded) == 0:
                            raise RuntimeError("Fine-Tuned WavLM evaluation could not read any audio files successfully.")
                        metrics = summarize_predictions(probabilities, y_test_encoded, label_encoder)
                        roc_data = compute_roc_curves_from_probabilities(probabilities, y_test_encoded, label_encoder)
                        sample_count = len(y_test_encoded)
                    else:
                        X_test, y_test = _load_feature_dataset_cached(tuple(active_paths))
                        if scope_mode == "Fast Snapshot":
                            X_test, y_test = _sample_feature_dataset(X_test, y_test, per_class_limit)
                        y_test_encoded = label_encoder.transform(y_test)
                        metrics = evaluate_model(model, X_test, y_test_encoded, label_encoder, scaler)
                        roc_data = compute_roc_curves(model, X_test, y_test_encoded, label_encoder, scaler)
                        sample_count = len(y_test_encoded)
                        audit_summary = None

                    st.session_state["eval_results"] = {
                        "metrics": metrics,
                        "roc_data": roc_data,
                        "model_name": model_name,
                        "protocol_mode": protocol_mode,
                        "scope_mode": scope_mode,
                        "sample_count": sample_count,
                        "audit_summary": audit_summary,
                        "y_true": y_test_encoded,
                    }
                    st.rerun()
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

    res_data = st.session_state["eval_results"]
    if res_data is None:
        st.markdown(
            """
            <div style='background:#0f1520;border:1px dashed #1e2d45;border-radius:14px;padding:60px;text-align:center;'>
                <div style='font-size:48px;margin-bottom:10px;'>📊</div>
                <div style='font-size:18px;font-weight:600;color:#f1f5f9;'>No evaluation results yet</div>
                <div style='font-size:14px;color:#64748b;margin-top:6px;'>Choose your scope and click 'Run Evaluation' to begin.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    metrics = res_data["metrics"]
    roc_data = res_data["roc_data"]
    y_true = np.asarray(res_data.get("y_true", []))

    st.caption(
        f"Protocol: {res_data.get('protocol_mode', 'Unknown')} | "
        f"Scope: {res_data.get('scope_mode', 'Unknown')} | "
        f"Samples evaluated: {res_data.get('sample_count', 'Unknown')}"
    )
    st.info(
        "Dashboard evaluation measures the selected folders directly. For research-grade benchmarking, "
        "use a held-out speaker or session split in the training scripts."
    )

    st.markdown("<div class='section-title'>Overall Performance</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    metrics_list = [
        ("Accuracy", f"{metrics['accuracy']*100:.2f}%", "overall correct", ACCENT),
        ("F1-Score", f"{metrics['f1']:.4f}", "weighted average", ACCENT2),
        ("Precision", f"{metrics['precision']:.4f}", "reliability", ACCENT_GREEN),
        ("Recall", f"{metrics['recall']:.4f}", "sensitivity", ACCENT_PINK),
    ]
    for col, (label, value, sub, color) in zip(cols, metrics_list):
        with col:
            st.markdown(
                f"""
                <div class='metric-card animated-border'>
                    <h3>{label}</h3>
                    <div class='value' style='color:{color};'>{value}</div>
                    <div class='sub'>{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("<div class='section-title'>Evaluation Insights</div>", unsafe_allow_html=True)
    confidence = _confidence_summary(metrics["probabilities"], metrics["predictions"], y_true) if len(y_true) else None
    per_class = metrics["per_class_metrics"]
    ranked_classes = per_class.sort_values(["F1-Score", "Support"], ascending=[False, False]).reset_index(drop=True)
    best_class = ranked_classes.iloc[0] if not ranked_classes.empty else None
    worst_class = ranked_classes.iloc[-1] if not ranked_classes.empty else None

    insight_cols = st.columns(4)
    insight_cards = [
        ("Avg Confidence", f"{confidence['avg_confidence']:.1f}%" if confidence else "n/a", "top-1 probability", ACCENT),
        ("Avg Margin", f"{confidence['avg_margin']:.1f}%" if confidence else "n/a", "top-1 vs top-2 gap", ACCENT2),
        (
            "Best Emotion",
            str(best_class["Emotion"]).capitalize() if best_class is not None else "n/a",
            f"F1 {best_class['F1-Score']:.2f}" if best_class is not None else "no data",
            ACCENT_GREEN,
        ),
        (
            "Needs Attention",
            str(worst_class["Emotion"]).capitalize() if worst_class is not None else "n/a",
            f"F1 {worst_class['F1-Score']:.2f}" if worst_class is not None else "no data",
            ACCENT_PINK,
        ),
    ]
    for col, (label, value, sub, color) in zip(insight_cols, insight_cards):
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

    if confidence:
        st.caption(
            f"Correct predictions average {confidence['correct_confidence']:.1f}% confidence, "
            f"incorrect ones average {confidence['incorrect_confidence']:.1f}%, and "
            f"{confidence['low_conf_share']:.1f}% of predictions fall below 60% confidence."
        )

    audit_summary = res_data.get("audit_summary") or []
    if audit_summary:
        st.markdown("**Speaker / session audit groups used**")
        st.dataframe(audit_summary, use_container_width=True, hide_index=True)

    confusion_pairs = _top_confusion_pairs(metrics["confusion_matrix"], metrics["classes"])
    if confusion_pairs:
        st.markdown("**Top confusion pairs**")
        st.dataframe(confusion_pairs, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
    cm = metrics["confusion_matrix"]
    classes = metrics["classes"]
    fig_cm, ax_cm = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax_cm,
        cbar=False,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig_cm)

    st.markdown("---")
    st.markdown("<div class='section-title'>Multi-Class ROC Curves</div>", unsafe_allow_html=True)
    fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
    for cls_name, data in roc_data.items():
        color = EMOTION_COLORS.get(cls_name.lower(), ACCENT)
        auc_val = data["auc"]
        auc_text = "N/A" if np.isnan(auc_val) else f"{auc_val:.2f}"
        ax_roc.plot(data["fpr"], data["tpr"], color=color, lw=2, label=f"{cls_name} (AUC={auc_text})")
    ax_roc.plot([0, 1], [0, 1], color="#374151", linestyle="--")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right", fontsize=10, frameon=False)
    plt.tight_layout()
    st.pyplot(fig_roc)

    st.markdown("---")
    st.markdown("<div class='section-title'>Per-Class Breakdown</div>", unsafe_allow_html=True)
    header_cols = st.columns([2, 1, 1, 1, 1, 3])
    for col, text in zip(header_cols, ["Emotion", "Precision", "Recall", "F1", "Support", "Score Dist"]):
        col.markdown(
            f"<div style='font-size:11px;color:#64748b;text-transform:uppercase;'>{text}</div>",
            unsafe_allow_html=True,
        )

    for _, row in per_class.iterrows():
        emotion = row["Emotion"]
        emoji = EMOTION_EMOJI.get(emotion.lower(), "🎭")
        color = EMOTION_COLORS.get(emotion.lower(), ACCENT)
        pct = int(row["F1-Score"] * 100)

        row_cols = st.columns([2, 1, 1, 1, 1, 3])
        row_cols[0].markdown(f"**{emoji} {emotion.capitalize()}**")
        row_cols[1].markdown(f"{row['Precision']:.2f}")
        row_cols[2].markdown(f"{row['Recall']:.2f}")
        row_cols[3].markdown(f"**{row['F1-Score']:.2f}**")
        row_cols[4].markdown(f"{int(row['Support'])}")
        row_cols[5].markdown(
            f"""
            <div class='diversity-bar-bg' style='margin-top:8px;'>
                <div class='diversity-bar-fill' style='width:{pct}%;background:linear-gradient(90deg,{color}66,{color});'></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("<div class='section-title'>Report Export</div>", unsafe_allow_html=True)
    try:
        pdf_bytes = generate_pdf_report(metrics, label_encoder)
        st.download_button(
            label="Download Performance Report (PDF)",
            data=pdf_bytes,
            file_name=f"emotion_ai_report_{model_name.replace(' ', '_').lower()}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"PDF generation unavailable: {exc}")

    if st.button("Clear Results", use_container_width=True):
        st.session_state["eval_results"] = None
        st.rerun()

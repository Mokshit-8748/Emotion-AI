"""Live Inference page — microphone + file upload, full visualizations."""

import hashlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from config import EMOTION_COLORS, EMOTION_EMOJI, ACCENT, ACCENT2, ACCENT_PINK, ACCENT_GREEN


def _dark_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d0f14")
    ax.set_facecolor("#111520")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a3350")
    ax.tick_params(colors="#4a5568")
    return fig, ax


def _run_inference(audio_bytes, model, label_encoder, scaler):
    """Extract features from audio bytes and return prediction."""
    import soundfile as sf
    from src.audio_preprocessing import preprocess_audio
    from src.feature_extraction import extract_features

    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    audio_proc = preprocess_audio(audio, sr)
    feats = extract_features(audio_proc, sr)
    feats = np.array(feats, dtype=np.float32)

    if scaler is not None:
        feats = scaler.transform(feats.reshape(1, -1)).flatten()

    feats_model = feats.reshape(1, feats.shape[0], 1)
    pred    = model(feats_model, training=False).numpy()
    idx     = int(np.argmax(pred))
    emotion = label_encoder.inverse_transform([idx])[0]
    conf    = float(np.max(pred)) * 100
    probs   = pred[0]
    labels  = label_encoder.classes_
    return audio, sr, feats, emotion, conf, probs, labels


def _run_wavlm_inference(audio_bytes, model):
    import soundfile as sf

    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    result = model.predict_audio(audio, sr)
    labels = np.array(result["labels"])
    probs = np.asarray(result["probabilities"], dtype=np.float32)
    return (
        result["audio"],
        result["sr"],
        np.zeros((result["input_samples"],), dtype=np.float32),
        result["emotion"],
        result["confidence"],
        probs,
        labels,
    )


def _prediction_cache_key(selected_model_name, audio_bytes):
    return f"{selected_model_name}:{hashlib.sha1(audio_bytes).hexdigest()}"


def _waveform_plot(audio, sr):
    fig, ax = _dark_fig(10, 2.5)
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.fill_between(t, audio, alpha=0.6, color=ACCENT)
    ax.plot(t, audio, color=ACCENT, linewidth=0.6, alpha=0.9)
    ax.axhline(0, color="#2a3350", linewidth=0.8)
    ax.set_xlabel("Time (s)", color="#7a8aaa", fontsize=9)
    ax.set_ylabel("Amplitude", color="#7a8aaa", fontsize=9)
    ax.set_title("Waveform", color="#e8e8e8", fontsize=10, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def _mel_plot(audio, sr):
    import librosa
    n_fft = min(1024, len(audio))
    hop   = n_fft // 4
    mel   = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64,
                                            n_fft=n_fft, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = _dark_fig(10, 3)
    img = ax.imshow(mel_db, aspect="auto", origin="lower",
                    cmap="magma", interpolation="nearest")
    plt.colorbar(img, ax=ax, format="%+2.0f dB",
                 label="dB").ax.yaxis.label.set_color("#7a8aaa")
    ax.set_xlabel("Frame", color="#7a8aaa", fontsize=9)
    ax.set_ylabel("Mel Band", color="#7a8aaa", fontsize=9)
    ax.set_title("Mel Spectrogram", color="#e8e8e8", fontsize=10, pad=8)
    plt.tight_layout()
    return fig


def _radar_chart(probs, labels):
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals   = (probs * 100).tolist()
    angles += angles[:1]; vals += vals[:1]
    fig = plt.figure(figsize=(5, 5), facecolor="#0d0f14")
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#111520")
    ax.plot(angles, vals, color=ACCENT, linewidth=2)
    ax.fill(angles, vals, color=ACCENT, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [f"{EMOTION_EMOJI.get(l,'')} {l}" for l in labels],
        fontsize=9, color="#e8e8e8",
    )
    ax.set_yticklabels([])
    ax.grid(color="#2a3350", linestyle="--", alpha=0.6)
    ax.spines["polar"].set_edgecolor("#2a3350")
    plt.tight_layout()
    return fig


def _confidence_gauge_html(conf):
    if conf >= 70:
        color = "#68d391"; label = "High Confidence"; bg = "#1a3a22"
    elif conf >= 50:
        color = "#f6ad55"; label = "Moderate Confidence"; bg = "#2a2010"
    else:
        color = "#fc8181"; label = "Low Confidence"; bg = "#2a1010"

    pct = min(100, conf)
    # CSS arc gauge using conic-gradient
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                padding:16px;border-radius:12px;background:{bg};
                border:1px solid {color}33;margin-bottom:12px;">
      <div style="position:relative;width:120px;height:120px;margin-bottom:8px;">
        <div style="
          width:120px;height:120px;border-radius:50%;
          background: conic-gradient({color} {pct*3.6:.1f}deg, #1e2433 0deg);
          display:flex;align-items:center;justify-content:center;">
          <div style="
            width:90px;height:90px;border-radius:50%;background:#0d0f14;
            display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <span style="font-size:22px;font-weight:800;color:{color};">{conf:.1f}%</span>
          </div>
        </div>
      </div>
      <span style="font-size:13px;font-weight:700;color:{color};letter-spacing:1px;">
        {label}
      </span>
    </div>"""


def _top3_bars_html(probs, labels):
    sorted_idx  = np.argsort(probs)[::-1][:3]
    html = "<div style='margin-top:8px;'>"
    for rank, i in enumerate(sorted_idx):
        lbl   = labels[i]
        pct   = probs[i] * 100
        color = EMOTION_COLORS.get(lbl, ACCENT)
        medal = ["🥇", "🥈", "🥉"][rank]
        html += f"""
        <div style='margin-bottom:10px;'>
          <div style='display:flex;justify-content:space-between;
                      font-size:13px;color:#e8e8e8;margin-bottom:3px;'>
            <span>{medal} {EMOTION_EMOJI.get(lbl,'')} {lbl.capitalize()}</span>
            <span style='font-family:monospace;color:{color};'>{pct:.1f}%</span>
          </div>
          <div style='background:#1a2035;border-radius:6px;height:8px;overflow:hidden;'>
            <div style='width:{pct:.1f}%;height:8px;border-radius:6px;
                        background:linear-gradient(90deg,{color}99,{color});
                        transition:width 0.6s ease;'></div>
          </div>
        </div>"""
    html += "</div>"
    return html


def _top2_summary(probs, labels):
    sorted_idx = np.argsort(probs)[::-1]
    primary_idx = int(sorted_idx[0])
    runner_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else primary_idx
    primary_label = str(labels[primary_idx])
    runner_label = str(labels[runner_idx])
    primary_pct = float(probs[primary_idx]) * 100.0
    runner_pct = float(probs[runner_idx]) * 100.0
    return {
        "primary_label": primary_label,
        "runner_label": runner_label,
        "primary_pct": primary_pct,
        "runner_pct": runner_pct,
        "margin_pct": max(0.0, primary_pct - runner_pct),
    }


def render(model, label_encoder, scaler, selected_model_name, model_err=None):
    st.markdown("# 🎙️ Live Speech Emotion Inference")
    st.markdown(
        f"Active model: <span style='color:#63b3ed;font-weight:800;'>"
        f"{selected_model_name}</span> — Upload a WAV file or record your voice.",
        unsafe_allow_html=True,
    )

    if model is None:
        detail = model_err or "Check model path in sidebar."
        st.error(f"SER model not loaded. {detail}")
        return
    if label_encoder is None:
        st.error("Label encoder not loaded. Check models/label_encoder.pkl.")
        return

    tab_upload, tab_mic = st.tabs(["📁 Upload File", "🎤 Record Audio"])

    audio_bytes = None
    if "inference_cache_key" not in st.session_state:
        st.session_state["inference_cache_key"] = None
    if "inference_cache_result" not in st.session_state:
        st.session_state["inference_cache_result"] = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Drop a WAV/FLAC/OGG file here", type=["wav", "flac", "ogg"],
            key="inf_upload",
        )
        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(audio_bytes, format="audio/wav")

    with tab_mic:
        st.markdown(
            "<div style='background:#111520;border:1px solid #2a3350;"
            "border-radius:10px;padding:14px;margin-bottom:12px;'>"
            "<b style='color:#63b3ed;'>🎤 Click below to record directly from your microphone.</b>"
            "<br><span style='color:#7a8aaa;font-size:12px;'>"
            "Allows browser mic access — works on localhost and HTTPS.</span></div>",
            unsafe_allow_html=True,
        )
        try:
            # ── Key-toggling is the only way to force a full widget UI reset ──
            if "mic_reset_id" not in st.session_state:
                st.session_state["mic_reset_id"] = 0

            if st.button("🗑️ Clear Recording", key="btn_clear_mic"):
                st.session_state["mic_reset_id"] += 1
                st.rerun()

            # Dynamic key forces Streamlit to treat this as a 'New' widget
            mic_key = f"inf_mic_{st.session_state['mic_reset_id']}"
            recorded = st.audio_input("Record audio", key=mic_key)
            
            if recorded:
                audio_bytes = recorded.read()
                st.audio(audio_bytes, format="audio/wav")
        except AttributeError:
            st.info("st.audio_input requires Streamlit ≥ 1.31. Upgrade: pip install -U streamlit")

    # ── Run inference ────────────────────────────────────────────────────────
    if audio_bytes:
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            run = st.button("🚀 Analyse Emotion", key="inf_run")

        if run:
            with st.spinner("Extracting features and running inference…"):
                try:
                    if getattr(model, "ser_backend", "keras_feature") == "wavlm_pt":
                        audio, sr, feats, emotion, conf, probs, labels = \
                            _run_wavlm_inference(audio_bytes, model)
                        feature_title = "Input Samples"
                        feature_sub = "16kHz raw audio window"
                    else:
                        audio, sr, feats, emotion, conf, probs, labels = \
                            _run_inference(audio_bytes, model, label_encoder, scaler)
                        feature_title = "Feature Dims"
                        feature_sub = "MFCC+chroma+mel+pitch…"
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    return

            st.markdown("---")
            top2 = _top2_summary(probs, labels)
            backend = getattr(model, "ser_backend", "keras_feature")

            # ── Metric cards row ─────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class='metric-card animated-border pulse-card' 
                     style='border-left: 4px solid {EMOTION_COLORS.get(emotion, ACCENT)};'>
                  <h3>Detected Emotion</h3>
                  <div class='value' style='color:{EMOTION_COLORS.get(emotion, ACCENT)};'>
                    {EMOTION_EMOJI.get(emotion,'')} {emotion.upper()}
                  </div>
                  <div class='sub'>top prediction</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                conf_col = ACCENT_GREEN if conf >= 70 else (ACCENT2 if conf >= 50 else ACCENT_PINK)
                st.markdown(f"""
                <div class='metric-card animated-border'>
                  <h3>Confidence</h3>
                  <div class='value' style='color:{conf_col};'>{conf:.1f}%</div>
                  <div class='sub'>model certainty</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class='metric-card animated-border'>
                  <h3>{feature_title}</h3>
                  <div class='value'>{feats.shape[0]}</div>
                  <div class='sub'>{feature_sub}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Waveform + Mel Spectrogram ────────────────────────────────
            st.markdown("### 📉 Audio Visualizations")
            fig_wave = _waveform_plot(audio, sr)
            st.pyplot(fig_wave); plt.close(fig_wave)

            fig_mel = _mel_plot(audio, sr)
            st.pyplot(fig_mel); plt.close(fig_mel)

            st.markdown("---")

            # ── Confidence gauge + Top-3 + Radar ─────────────────────────
            st.markdown("### 🎯 Prediction Breakdown")
            left, mid, right = st.columns([1, 1, 1])

            with left:
                st.markdown("**Confidence Meter**")
                st.markdown(_confidence_gauge_html(conf), unsafe_allow_html=True)

            with mid:
                st.markdown("**Top 3 Emotions**")
                st.markdown(_top3_bars_html(probs, labels), unsafe_allow_html=True)

            with right:
                st.markdown("**Emotion Radar**")
                fig_radar = _radar_chart(probs, labels)
                st.pyplot(fig_radar); plt.close(fig_radar)

            # ── All probabilities bar chart ───────────────────────────────
            st.markdown("### 📊 Full Probability Distribution")
            fig_bar, ax = _dark_fig(10, 3.5)
            bar_colors = [
                EMOTION_COLORS.get(lbl, ACCENT) if lbl == emotion else "#2a3a55"
                for lbl in labels
            ]
            bars = ax.bar(labels, probs * 100, color=bar_colors,
                          edgecolor="#1e2433", linewidth=0.8, width=0.55)
            for bar, p in zip(bars, probs * 100):
                if p > 1:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.8, f"{p:.1f}%",
                            ha="center", va="bottom", fontsize=9, color="#e8e8e8")
            ax.set_ylabel("Probability (%)", fontsize=10)
            ax.set_ylim(0, 115)
            ax.set_title("Prediction Confidence per Emotion",
                         fontsize=11, pad=12, color="#e8e8e8")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_bar); plt.close(fig_bar)

            # ── Confidence interpretation ─────────────────────────────────
            st.markdown("---")
            if conf >= 70:
                st.success(f"✅ High confidence — model is certain about **{emotion}**.")
            elif conf >= 50:
                st.warning(f"⚠️ Moderate confidence ({conf:.1f}%). Try the Fine-Tuned WavLM model.")
            else:
                st.error(f"❌ Low confidence ({conf:.1f}%). Ambiguous sample or noise.")

            if conf < 70:
                action_text = (
                    "Switch to Fine-Tuned WavLM for stronger accuracy on difficult clips."
                    if backend != "wavlm_pt"
                    else "Retry with a cleaner 2-4 second recording and less background noise."
                )
                advice_cols = st.columns([1.25, 1, 1, 1])
                with advice_cols[0]:
                    st.markdown(
                        f"""
                        <div class='boost-card' style='border-left-color:{ACCENT2};'>
                          <div class='rank' style='color:{ACCENT2};'>Confidence Guidance</div>
                          <div class='title'>Top-2 reading</div>
                          <div class='desc'>
                            Primary: <b>{EMOTION_EMOJI.get(top2['primary_label'], '')} {top2['primary_label'].capitalize()}</b>
                            ({top2['primary_pct']:.1f}%)<br>
                            Runner-up: <b>{EMOTION_EMOJI.get(top2['runner_label'], '')} {top2['runner_label'].capitalize()}</b>
                            ({top2['runner_pct']:.1f}%)<br>
                            Margin: <b>{top2['margin_pct']:.1f}%</b><br>
                            Recommendation: {action_text}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with advice_cols[1]:
                    st.markdown(
                        f"""
                        <div class='metric-card animated-border'>
                          <h3>Top Choice</h3>
                          <div class='value' style='font-size:24px;color:{EMOTION_COLORS.get(top2['primary_label'], ACCENT)};'>
                            {top2['primary_pct']:.1f}%
                          </div>
                          <div class='sub'>{top2['primary_label']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with advice_cols[2]:
                    st.markdown(
                        f"""
                        <div class='metric-card animated-border'>
                          <h3>Runner-Up</h3>
                          <div class='value' style='font-size:24px;color:{EMOTION_COLORS.get(top2['runner_label'], ACCENT2)};'>
                            {top2['runner_pct']:.1f}%
                          </div>
                          <div class='sub'>{top2['runner_label']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with advice_cols[3]:
                    st.markdown(
                        f"""
                        <div class='metric-card animated-border'>
                          <h3>Margin</h3>
                          <div class='value' style='font-size:24px;color:{ACCENT2};'>{top2['margin_pct']:.1f}%</div>
                          <div class='sub'>top-1 vs top-2 gap</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if emotion.lower() == "happy":
                st.balloons()

"""
app.py  —  OCT Retinal Disease Classifier
Run with:  streamlit run app.py
"""

import os
import torch
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import gdown

from model.loader    import load_config, load_model, count_parameters, DEVICE
from model.predictor import (
    preprocess_image,
    predict_image,
    predict_with_uncertainty,
    generate_gradcam,
    is_ood,
    ENTROPY_THRESHOLD,
    STD_THRESHOLD,
)
from utils.visualization import (
    plot_confidence_bars,
    plot_preprocessing_steps,
    build_gradcam_figure,
    plot_uncertainty,
    pil_to_thumbnail,
)

# ── 1. Page config (DEVE essere il primissimo comando Streamlit) ───────────────
st.set_page_config(
    page_title="ORCA — OCT Retinal Classification Assistant",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Paths ──────────────────────────────────────────────────────────────────
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
CONFIG_PATH = os.path.join(WEIGHTS_DIR, "config.json")

# ── 3. Download dei pesi se mancanti ───────────────────────────────────────────
MODELS_TO_DOWNLOAD = {
    "best_CustomCNN.pth": "1c5TPlxQOIaqwME9i0gK5NikLVoTT9rWu",
    "best_EfficientNet.pth": "1j1S3RSsVs0P9IP6JNsB7O4La9w8kNIVN",
    "best_MedGemma.pth": "1AN30OH4QvevAB2JTZxI7g3CPUH_Cr2Ck",
    "MIRAGE_Base_OCT5k.pth": "1MmnGaqS74VLnHDOCVdcI1kpOZD33u5NU",
}

@st.cache_resource
def download_weights_if_missing():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for filename, file_id in MODELS_TO_DOWNLOAD.items():
        destination = os.path.join(WEIGHTS_DIR, filename)
        if not os.path.exists(destination):
            with st.spinner(f"(Downloading weights {filename})..."):
                gdown.download(id=file_id, output=destination, quiet=False)

download_weights_if_missing()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; color: #2D3142; }
h1, h2, h3                  { font-family: 'Plus Jakarta Sans', sans-serif; }
[data-testid="stMetricValue"]{ font-family: 'Roboto Mono', monospace !important;
                               font-size: 1.3rem !important; color: #2D3142 !important; }
[data-testid="stMetricLabel"]{ font-size: 0.76rem !important; color: #6B7280 !important;
                               text-transform: uppercase; letter-spacing: 0.04em; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }

img { border-radius: 12px; }

.stButton > button {
    border-radius: 12px; background-color: #84A59D;
    color: white; border: none;
    font-family: 'Inter', sans-serif; font-weight: 600;
    padding: 0.5rem 1.5rem; transition: background 0.2s;
}
.stButton > button:hover { background-color: #6D8E86; color: white; }

.card {
    background: #FFFFFF; border: 1px solid #E8E4DF;
    border-radius: 12px; padding: 20px 24px;
    box-shadow: 0 2px 10px rgba(45,49,66,0.07); margin-bottom: 14px;
}
.result-AMD    { background:#FEF0F0; border-left:5px solid #F28482;
                 border-radius:12px; padding:18px 22px; }
.result-DME    { background:#FEF5EE; border-left:5px solid #E29578;
                 border-radius:12px; padding:18px 22px; }
.result-Normal { background:#EEF5F3; border-left:5px solid #84A59D;
                 border-radius:12px; padding:18px 22px; }

.ood-critical {
    background:#FEF0F0; border:1px solid #F28482;
    border-radius:10px; padding:14px 18px; color:#7A2828; margin:10px 0;
}
.ood-warning {
    background:#FEF8EC; border:1px solid #E29578;
    border-radius:10px; padding:14px 18px; color:#7A4010; margin:10px 0;
}
.ood-ok {
    background:#EEF5F3; border:1px solid #84A59D;
    border-radius:10px; padding:14px 18px; color:#2A4A45; margin:10px 0;
}
.signal-badge {
    display:inline-block; border-radius:6px; padding:2px 10px;
    font-size:0.78rem; font-weight:600; margin:2px 4px 2px 0;
}
.badge-fail { background:#FEF0F0; color:#C0392B; }
.badge-ok   { background:#EEF5F3; color:#2A6349; }

.disclaimer {
    position:fixed; bottom:0; left:0; width:100%;
    background:#F0EDE8; border-top:1px solid #DDD8D0;
    color:#6B7280; font-size:0.76rem; padding:7px 24px; z-index:999;
    text-align: center;        /* ← aggiunta */
}
.main .block-container { padding-bottom: 52px; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
</style>
<div class="disclaimer">
This tool is intended for research purposes only and provides decision support.
It does not replace the judgment of a qualified ophthalmologist.
</div>
""", unsafe_allow_html=True)

# ── Load config ───────────────────────────────────────────────────────────────
config      = load_config(CONFIG_PATH)
CLASS_NAMES = config["class_names"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("logo.png", width=150)
    st.markdown(
        "<h2 style='font-family:Plus Jakarta Sans;font-size:1.15rem;"
        "color:#2D3142;margin-bottom:2px'>ORCA</h2>"
        "<p style='color:#84A59D;font-size:0.78rem;margin-top:0;margin-bottom:1px'>"
        "OCT Retinal Classification Assistant</p>"
        "<p style='color:#B0B7C3;font-size:0.72rem;margin-top:0'>"
        "v1.0 · Research Prototype</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    
   

    st.markdown("**Upload an OCT image**")
    uploaded_file = st.file_uploader(
        "PNG, JPG, TIFF",
        type=["png", "jpg", "jpeg", "tiff", "tif"],
        label_visibility="collapsed",
    )
    st.divider()

    model_name = st.selectbox(
        "Model",
        options=["CustomCNN", "EfficientNet", "MIRAGE", "MedGemma"],
        index=1,
        help=(
            "CustomCNN: lightweight 3-layer baseline trained from scratch "
            "(test accuracy: 85.4%). "
            "EfficientNet: CNN with transfer learning from ImageNet "
            "(test accuracy: 95.7%). "
            "MIRAGE: foundation model. A Vision Transformer (ViT-Base) pre-trained "
            "on large scale multi-modal retinal data "
            "(test balanced accuracy: 96.0%, AUC: 0.995 on OCT5k). "
            "MedGemma: foundation model. A frozen SigLIP-So400m vision encoder "
            "(google/siglip-so400m-patch14-384) pre-trained on large scale medical "
            "image data, with a fine-tuned MLP classification head "
            "(test accuracy: 90.0% on OCT5k). "
            "Unlike CustomCNN and EfficientNet, which learn everything from the "
            "OCT5k dataset alone, MIRAGE and MedGemma start from a large model "
            "already pre-trained on huge amounts of data, and only a small "
            "classification head is trained on top for this task. "
            "Downloads ~3.5 GB on first use, then cached locally."
        ),
    )

    ood_threshold = st.slider(
        "Confidence threshold",
        min_value=0.50, max_value=0.95, value=0.65, step=0.05,
        help="Predictions below this value trigger a confidence alert.",
    )

    use_mc = st.toggle(
        "Enable MC Dropout uncertainty",
        value=True,
        help=(
            "Runs 20 stochastic forward passes to estimate prediction uncertainty. "
            "More robust OOD detection. Adds ~1 second to inference time. "
            "Compatible with all models. For MedGemma, stochasticity comes from "
            "the MLP head dropout (the frozen vision encoder is deterministic)."
        ),
    )


    st.divider()
    with st.expander("About this application"):
        st.markdown(
            "Research prototype for automated classification of retinal OCT scans "
            "into AMD, DME, or Normal. Four model architectures are available: "
            "CustomCNN (lightweight baseline), EfficientNet-B0 (CNN with transfer learning), "
            "MIRAGE (Vision Transformer pre-trained on multi-modal retinal data), and "
            "MedGemma (frozen SigLIP-So400m vision encoder with a fine-tuned MLP head). "
            "OOD detection uses MC Dropout (Gal & Ghahramani, 2016) "
            "and Shannon entropy to flag images outside the training distribution. "
            "MIRAGE and MedGemma saliency maps use Attention Rollout (Abnar & Zuidema, 2020)."
        )

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {model_name}..."):
    model, is_real = load_model(model_name, WEIGHTS_DIR)

st.sidebar.caption("Trained weights loaded" if is_real else "Demo mode (random predictions)")

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:1.85rem;margin-bottom:2px'>ORCA</h1>"
    "<p style='color:#84A59D;font-size:0.88rem;font-weight:600;margin-top:0;margin-bottom:6px'>"
    "OCT Retinal Classification Assistant</p>"
    "<p style='color:#6B7280;font-size:0.92rem;margin-top:0'>"
    "Upload a retinal OCT scan from the sidebar. "
    "The classifier returns a predicted diagnosis, confidence scores, "
    "and uncertainty estimates.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Main columns ──────────────────────────────────────────────────────────────
pil_image  = None
tensor     = None
steps_imgs = None

col_img, col_panel = st.columns([7, 3], gap="large")

with col_img:
    if uploaded_file is None:
        st.markdown(
            "<div class='card' style='min-height:320px;display:flex;"
            "align-items:center;justify-content:center;color:#9499A8;"
            "font-size:0.95rem;text-align:center'>"
            "Upload an OCT image from the sidebar to begin.</div>",
            unsafe_allow_html=True,
        )
    else:
        try:
            pil_image = Image.open(uploaded_file)
            pil_image.verify()
            uploaded_file.seek(0)
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Uploaded OCT scan", use_container_width=True)
            tensor, steps_imgs = preprocess_image(pil_image, model_name=model_name)
        except Exception:
            st.error(
                "The uploaded file does not appear to be a valid OCT image. "
                "Please upload a PNG, JPG, or TIFF file."
            )

with col_panel:
    if pil_image is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-family:Plus Jakarta Sans;font-weight:600;"
            "font-size:0.82rem;color:#6B7280;text-transform:uppercase;"
            "letter-spacing:0.05em;margin-bottom:10px'>Image details</p>",
            unsafe_allow_html=True,
        )
        w, h = pil_image.size
        c1, c2 = st.columns(2)
        c1.metric("Width",  f"{w} px")
        c2.metric("Height", f"{h} px")
        c3, c4 = st.columns(2)
        c3.metric("Format", pil_image.format or "N/A")
        c4.metric("Mode",   pil_image.mode)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if pil_image is not None and tensor is not None:
        predict_clicked = st.button("Run prediction", type="primary", use_container_width=True)
    else:
        predict_clicked = False

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-family:Plus Jakarta Sans;font-weight:600;"
        "font-size:0.82rem;color:#6B7280;text-transform:uppercase;"
        "letter-spacing:0.05em;margin-bottom:10px'>Model info</p>",
        unsafe_allow_html=True,
    )
    st.metric("Architecture", model_name)
    st.metric("Device",       str(DEVICE).upper())
    n_par = count_parameters(model)
    st.metric("Parameters",   f"{n_par:,}" if n_par > 1 else "N/A")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Preprocessing expander ────────────────────────────────────────────────────
if steps_imgs is not None:
    with st.expander("Preprocessing pipeline (AI Act transparency)", expanded=False):
        st.caption(
            "Steps applied identically at training and inference time. "
            "Shown in compliance with EU AI Act Article 13."
        )
        fig_pre = plot_preprocessing_steps(steps_imgs)
        st.pyplot(fig_pre, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════════════

if predict_clicked:

    passes_label = "Running 20 MC Dropout passes..." if use_mc else "Running inference..."
    with st.spinner(passes_label):
        if use_mc:
            result = predict_with_uncertainty(model, tensor, CLASS_NAMES, DEVICE, n_passes=20)
        else:
            result = predict_image(model, tensor, CLASS_NAMES, DEVICE)

    pred_class = result["predicted_class"]
    pred_idx   = result["predicted_idx"]
    probs      = result["probabilities"]
    confidence = result["confidence"]
    entropy    = result["entropy"]
    mean_std   = result["mean_std"]
    inf_time   = result["inference_time_s"]
    flagged, reasons = is_ood(result, ood_threshold)

    st.divider()

    # ── OOD safety block ──────────────────────────────────────────────────────
    import numpy as _np
    max_entropy = _np.log(3)   # 1.099 nats

    n_signals  = len(reasons)
    card_class = "ood-critical" if n_signals >= 2 else ("ood-warning" if n_signals == 1 else "ood-ok")

    if flagged:
        badge_html = "".join(
            f"<span class='signal-badge badge-fail'>{r.split(':')[0]}</span>"
            for r in reasons
        )
        st.markdown(
            f"<div class='{card_class}'>"
            f"<strong>Uncertain prediction, human review recommended.</strong> "
            f"This image may show a pathology outside the training distribution "
            f"(e.g. glaucoma or a scan artefact). The following signals were triggered: "
            f"<br>{badge_html}"
            f"<br><small>Please refer this case to a qualified specialist before "
            f"drawing any clinical conclusions.</small></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='ood-ok'>"
            f"All uncertainty checks passed. "
            f"Confidence: {confidence * 100:.1f}% — "
            f"Entropy: {entropy:.3f} nats — "
            f"{'MC std: ' + f'{mean_std:.3f}' if use_mc else 'Single-pass mode'}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    res_left, res_right = st.columns([7, 3], gap="large")

    with res_left:
        st.image(pil_image, use_container_width=True)
        with st.expander("View evidence — Saliency map explanation", expanded=False):
            with st.spinner("Generating saliency map..."):
                heatmap = generate_gradcam(model, tensor, model_name, DEVICE, pred_idx)
            if heatmap is not None:
                fig_cam = build_gradcam_figure(pil_image, heatmap)
                st.pyplot(fig_cam, use_container_width=True)
                if model_name == "MIRAGE":
                    st.caption(
                        "Attention Rollout (Abnar & Zuidema, 2020): brighter regions show "
                        "where the Vision Transformer focused most when making its prediction. "
                        "The map is computed by recursively multiplying attention matrices "
                        "across all 12 transformer encoder layers."
                    )
                elif model_name == "MedGemma":
                    st.caption(
                        "Attention Rollout on the SigLIP-So400m vision encoder "
                        "(Abnar & Zuidema, 2020): brighter regions indicate patches that received "
                        "the most cumulative attention across all encoder layers. "
                        "The encoder processes the image at 384×384 px; "
                        "the map is upsampled to 512×512 for display."
                    )
                else:
                    st.caption(
                        "Grad-CAM: warmer colours indicate regions that most influenced the prediction. "
                        "For uncertain predictions, inspect whether the highlighted area "
                        "is anatomically consistent with the predicted class."
                    )
            else:
                st.info("Saliency map could not be generated for this configuration.")

    with res_right:
        label_colors = {"AMD": "#C0392B", "DME": "#B5651D", "Normal": "#2A6349"}
        lc = label_colors.get(pred_class, "#2D3142")

        st.markdown(
            f"<div class='result-{pred_class}'>"
            f"<p style='font-size:0.75rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.07em;color:{lc};margin:0 0 4px'>Predicted class</p>"
            f"<p style='font-family:Plus Jakarta Sans;font-size:2.3rem;"
            f"font-weight:700;color:{lc};margin:0 0 4px'>{pred_class}</p>"
            #f"<p style='font-family:Roboto Mono,monospace;font-size:1rem;"
            #f"color:{lc};margin:0'>Confidence: {confidence * 100:.1f}%</p>"
            f"<p style='font-size:0.76rem;color:#6B7280;margin-top:8px'>"
            f"Inference: {inf_time * 1000:.0f} ms"
            f"{'  |  ' + str(result['n_passes']) + ' MC passes' if use_mc else ''}"
            f"</p></div>",
            unsafe_allow_html=True,
        )

        if confidence >= 0.80:
            band, band_desc = "HIGH", "Strong AI evidence."
        elif confidence >= 0.55:
            band, band_desc = "MODERATE", "Supportive evidence, confirm with additional imaging."
        else:
            band, band_desc = "LOW", "Borderline prediction, clinical judgment required."

        st.markdown(
            f"<p style='font-size:0.82rem;color:{lc};margin:6px 0 0 0'>"
            f"AI confidence level: <strong>{band}</strong> — {band_desc}<br>"
            f"<span style='font-size:0.78rem;color:#6B7280'>"
            f"{'MIRAGE: foundation model, ViT-Base pre-trained on multi-modal retinal data (test BAcc 96.0%, AUC 0.995 on OCT5k).' if model_name == 'MIRAGE' else ('MedGemma: foundation model, frozen SigLIP-So400m encoder + fine-tuned MLP head (test accuracy 90.0% on OCT5k).' if model_name == 'MedGemma' else 'Based on a model trained on 117 OCT5k patients (AMD / DME / Normal).')}"
            f"</span></p>",
            unsafe_allow_html=True,
        )

        if not flagged:
            _next_action = {
                "AMD": "Findings suggestive of AMD. Clinical correlation and specialist review recommended.",
                "DME": "Findings suggestive of DME. Clinical correlation and specialist review recommended.",
                "Normal": "No signs of AMD or DME detected. Routine follow-up as clinically indicated.",
            }.get(pred_class, "Clinical correlation and specialist review recommended.")

            st.markdown(
                f"<div style='background:#F8F6F2;border-left:3px solid #84A59D;"
                f"border-radius:6px;padding:12px 16px;margin:12px 0 4px 0'>"
                f"<p style='font-size:0.75rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:0.06em;color:#6B7280;margin:0 0 4px'>Recommended next action</p>"
                f"<p style='font-size:0.88rem;color:#2D3142;margin:0'>{_next_action}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Class probabilities")
        st.pyplot(plot_confidence_bars(probs, CLASS_NAMES, pred_idx), use_container_width=True)

        # Uncertainty metrics
        if use_mc:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Uncertainty metrics")
            u1, u2 = st.columns(2)
            u1.metric(
                "Entropy",
                f"{entropy:.3f}",
                help=f"Shannon entropy (nats). Max possible: {max_entropy:.3f}. "
                     f"Flag threshold: {ENTROPY_THRESHOLD}.",
            )
            u2.metric(
                "MC std",
                f"{mean_std:.3f}",
                help=f"Mean standard deviation across 20 MC passes. "
                     f"Flag threshold: {STD_THRESHOLD}.",
            )
            # Mini uncertainty bar
            fig_unc = plot_uncertainty(result)
            if fig_unc is not None:
                st.pyplot(fig_unc, use_container_width=True)

    # ── Save to history ───────────────────────────────────────────────────────
    st.session_state.history.insert(0, {
        "thumbnail":  pil_to_thumbnail(pil_image),
        "filename":   uploaded_file.name,
        "model":      model_name,
        "class":      pred_class,
        "confidence": f"{confidence * 100:.1f}%",
        "entropy":    f"{entropy:.3f}",
        "ood_flag":   flagged,
        "time_ms":    f"{inf_time * 1000:.0f} ms",
    })
    st.session_state.history = st.session_state.history[:20]

# ── Session history ───────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    h1, h2 = st.columns([6, 1])
    with h1:
        st.markdown(
            "<h3 style='font-family:Plus Jakarta Sans;font-size:1rem;"
            "color:#2D3142'>Session history</h3>",
            unsafe_allow_html=True,
        )
    with h2:
        if st.button("Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    cols = st.columns(min(5, len(st.session_state.history)))
    for i, e in enumerate(st.session_state.history[:5]):
        with cols[i]:
            st.image(e["thumbnail"], use_container_width=True)
            flag = "(uncertain) " if e["ood_flag"] else ""
            st.caption(
                f"**{flag}{e['class']}** {e['confidence']}\n"
                f"H={e['entropy']} — {e['model']}"
            )

# ── Technical details ─────────────────────────────────────────────────────────
st.divider()
with st.expander("Technical details"):
    d1, d2, d3 = st.columns(3)
    d1.metric("Device",     str(DEVICE).upper())
    d2.metric("Model",      model_name)
    n_par = count_parameters(model)
    d3.metric("Parameters", f"{n_par:,}" if n_par > 1 else "N/A")
    st.caption(
        f"Weights: `{os.path.abspath(WEIGHTS_DIR)}`  \n"
        f"Config: `{CONFIG_PATH}`  \n"
        f"Input: 512x512 px, mean=0.5, std=0.5  \n"
        f"OOD signals: confidence < {ood_threshold:.0%}, "
        f"entropy > {ENTROPY_THRESHOLD}, MC std > {STD_THRESHOLD}"
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.78rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.05em;color:#6B7280;margin-bottom:6px'>Model card</p>",
        unsafe_allow_html=True,
    )
    if model_name == "MIRAGE":
        st.markdown(
            f"| Field | Value |\n|---|---|\n"
            f"| Architecture | MIRAGE (Vision Transformer, ViT-Base) |\n"
            f"| Patch size | 32 × 32 px (16 × 16 grid on 512 px input) |\n"
            f"| Parameters | ~86 M |\n"
            f"| Model type | Foundation model (large pre-trained backbone) |\n"
            f"| Training dataset | OCT5k (AMD / DME / Normal) |\n"
            f"| Reported test BAcc | 96.0% |\n"
            f"| Reported test AUC | 0.995 |\n"
            f"| Reported test MCC | 0.933 |\n"
            f"| Supported classes | AMD, DME, Normal |\n"
            f"| Out-of-scope inputs | Glaucoma, CNV, scan artefacts, paediatric scans |\n"
            f"| Saliency method | Attention Rollout (Abnar & Zuidema, 2020) |\n"
            f"| OOD signals | Confidence < {ood_threshold:.0%}, "
            f"Entropy > {ENTROPY_THRESHOLD}, MC std > {STD_THRESHOLD} |\n"
            f"| Intended use | Research prototype — decision support only |"
        )
    elif model_name == "MedGemma":
        st.markdown(
            f"| Field | Value |\n|---|---|\n"
            f"| Architecture | MedGemma (SigLIP-So400m encoder + MLP head) |\n"
            f"| Vision encoder | google/siglip-so400m-patch14-384 (public, frozen) |\n"
            f"| Embedding dim | 1152 (SigLIP pooler_output) |\n"
            f"| MLP head | Linear(1152→512) → LayerNorm → ReLU → Dropout → Linear(512→3) |\n"
            f"| Head parameters | 592,899 (trainable) |\n"
            f"| Model type | Foundation model (frozen pre-trained encoder) |\n"
            f"| Reported test accuracy | 90.0% |\n"
            f"| Input resolution | 384 × 384 px, RGB, normalised to [-1, 1] |\n"
            f"| Training dataset | OCT5k (AMD / DME / Normal) — head only |\n"
            f"| Supported classes | AMD, DME, Normal |\n"
            f"| Out-of-scope inputs | Glaucoma, CNV, scan artefacts, paediatric scans |\n"
            f"| Saliency method | Attention Rollout on SigLIP encoder (Abnar & Zuidema, 2020) |\n"
            f"| OOD signals | Confidence < {ood_threshold:.0%}, "
            f"Entropy > {ENTROPY_THRESHOLD}, MC std > {STD_THRESHOLD} |\n"
            f"| First-run download | ~3.5 GB (cached in ~/.cache/huggingface/hub/) |\n"
            f"| Intended use | Research prototype — decision support only |"
        )
    elif model_name == "CustomCNN":
        st.markdown(
            f"| Field | Value |\n|---|---|\n"
            f"| Model type | Trained from scratch (not a foundation model) |\n"
            f"| Reported test accuracy | 85.4% |\n"
            f"| Training dataset | OCT5k (AMD / DME / Normal) |\n"
            f"| Supported classes | AMD, DME, Normal |\n"
            f"| Out-of-scope inputs | Glaucoma, CNV, scan artefacts, paediatric scans |\n"
            f"| OOD signals | Confidence < {ood_threshold:.0%}, "
            f"Entropy > {ENTROPY_THRESHOLD}, MC std > {STD_THRESHOLD} |\n"
            f"| Intended use | Research prototype — decision support only |"
        )
    else:
        st.markdown(
            f"| Field | Value |\n|---|---|\n"
            f"| Model type | Transfer learning (not a foundation model) |\n"
            f"| Reported test accuracy | 95.7% |\n"
            f"| Training dataset | OCT5k (AMD / DME / Normal) |\n"
            f"| Supported classes | AMD, DME, Normal |\n"
            f"| Out-of-scope inputs | Glaucoma, CNV, scan artefacts, paediatric scans |\n"
            f"| OOD signals | Confidence < {ood_threshold:.0%}, "
            f"Entropy > {ENTROPY_THRESHOLD}, MC std > {STD_THRESHOLD} |\n"
            f"| Intended use | Research prototype — decision support only |"
        )
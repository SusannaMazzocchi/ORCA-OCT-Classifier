"""
pages/1_Information.py
-----------------------
Educational page covering the clinical context, model details,
and regulatory framework for non-expert users.
"""

import streamlit as st

st.set_page_config(
    page_title="Clinical Background | ORCA",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #2D3142; }
h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif; color: #2D3142; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }

img { border-radius: 12px; }

.card {
    background: #FFFFFF;
    border: 1px solid #E8E4DF;
    border-radius: 12px;
    padding: 24px 28px;
    box-shadow: 0 2px 10px rgba(45,49,66,0.07);
    margin-bottom: 18px;
}
.tag-amd    { background:#FEF0F0; color:#C0392B; border-radius:6px;
              padding:3px 10px; font-size:0.8rem; font-weight:600; }
.tag-dme    { background:#FEF5EE; color:#B5651D; border-radius:6px;
              padding:3px 10px; font-size:0.8rem; font-weight:600; }
.tag-normal { background:#EEF5F3; color:#2A6349; border-radius:6px;
              padding:3px 10px; font-size:0.8rem; font-weight:600; }
.section-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: #84A59D; margin-bottom: 6px;
}
.disclaimer {
    position: fixed; bottom: 0; left: 0; width: 100%;
    background: #F0EDE8; border-top: 1px solid #DDD8D0;
    color: #6B7280; font-size: 0.76rem;
    padding: 7px 24px; z-index: 999;
}
.main .block-container { padding-bottom: 52px; }
</style>
<div class="disclaimer">
This tool is intended for research purposes only and provides decision support.
It does not replace the judgment of a qualified ophthalmologist.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='font-size:1.9rem;margin-bottom:0'>Clinical Background</h1>"
    "<p style='color:#6B7280;font-size:0.93rem;margin-top:4px'>"
    "An overview of the retinal pathologies addressed by this classifier, "
    "the role of OCT in ophthalmology, and the regulatory context "
    "governing AI-based medical decision support systems.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# What is OCT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='section-label'>Imaging technology</div>", unsafe_allow_html=True)
st.markdown(
    "<h2 style='font-size:1.35rem;margin-top:0'>Optical Coherence Tomography</h2>",
    unsafe_allow_html=True,
)

st.markdown("""
<div class='card'>

Optical Coherence Tomography (OCT) is a non-invasive imaging modality that produces
high-resolution, cross-sectional images of the retina using near-infrared light.
The technique operates on the same physical principle as ultrasound, but achieves
a spatial resolution on the order of a few micrometers, allowing individual retinal
layers to be visualised in vivo.

OCT has become the gold standard for diagnosing and monitoring a range of retinal
conditions, including age-related macular degeneration and diabetic macular edema.
Its clinical adoption is extensive: it is used routinely in ophthalmology clinics
worldwide for screening, treatment planning, and post-treatment follow-up.

Despite its diagnostic power, the interpretation of OCT scans requires considerable
specialist expertise. Combined with increasing scan volumes driven by the growing
prevalence of age-related and metabolic diseases, there is a recognised need for
automated screening tools capable of flagging scans that require urgent clinical
review. This classifier was developed in that context.

</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# AMD
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='section-label'>Pathology</div>", unsafe_allow_html=True)
col_h, _ = st.columns([6, 2])
with col_h:
    st.markdown(
        "<h2 style='font-size:1.35rem;margin-top:0;display:inline'>Age-Related Macular Degeneration</h2>"
        "&nbsp;&nbsp;<span class='tag-amd'>AMD</span>",
        unsafe_allow_html=True,
    )

amd_col, amd_fact = st.columns([2, 1], gap="large")

with amd_col:
    st.markdown("""
<div class='card'>

Age-related macular degeneration is a progressive degenerative disease affecting the
macula, the central region of the retina responsible for high-acuity and colour vision.
It represents the leading cause of irreversible central vision loss in individuals over
the age of 60 in high-income countries, with an estimated global prevalence of
approximately 200 million people.

The condition is broadly categorised into two forms. Dry AMD, which accounts for
roughly 85 to 90 per cent of cases, is characterised by the gradual accumulation of
extracellular deposits known as drusen beneath the retinal pigment epithelium, followed
by progressive atrophy of photoreceptors and supporting cells. Although dry AMD advances
slowly, it can eventually lead to geographic atrophy and significant central vision loss.
Wet AMD, though less prevalent, is responsible for the majority of severe vision loss
associated with the disease. It involves the growth of abnormal choroidal blood vessels
that penetrate the retinal pigment epithelium and leak fluid or blood into the subretinal
space, causing rapid and irreversible photoreceptor damage if left untreated.

On OCT, AMD presents with characteristic structural signatures: drusen appear as
elevations beneath the retinal pigment epithelium, geographic atrophy is visible as
thinning and disruption of the outer retinal layers, and wet AMD produces hyperreflective
material and fluid accumulations that are readily identifiable by trained readers.
Early identification of these features is clinically significant because therapeutic
intervention with intravitreal anti-VEGF agents can substantially slow or stabilise
the progression of wet AMD, particularly when initiated before extensive photoreceptor
loss has occurred.

</div>
    """, unsafe_allow_html=True)

with amd_fact:
    st.info("""
**Epidemiological context**

Estimated global prevalence: 200 million individuals.

Leading cause of irreversible vision loss in adults over 60 in developed countries.

Dry AMD accounts for approximately 85 to 90 per cent of all cases.

Wet AMD, though less common, is responsible for the majority of cases involving
severe functional vision loss.

First-line treatment for wet AMD: intravitreal anti-VEGF injections.
    """)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# DME
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='section-label'>Pathology</div>", unsafe_allow_html=True)
col_h2, _ = st.columns([6, 2])
with col_h2:
    st.markdown(
        "<h2 style='font-size:1.35rem;margin-top:0;display:inline'>Diabetic Macular Edema</h2>"
        "&nbsp;&nbsp;<span class='tag-dme'>DME</span>",
        unsafe_allow_html=True,
    )

dme_col, dme_fact = st.columns([2, 1], gap="large")

with dme_col:
    st.markdown("""
<div class='card'>

Diabetic macular edema is a complication of diabetic retinopathy, itself a microvascular
complication of diabetes mellitus affecting the retinal vasculature. Chronically elevated
blood glucose levels damage the endothelial cells lining the retinal capillaries,
compromising the blood-retinal barrier and causing plasma and lipid exudates to leak
into the surrounding retinal tissue. When this leakage occurs in or near the macula,
the resulting accumulation of fluid produces retinal thickening and disrupts the
architectural organisation of the macular layers, leading to a degradation of central
visual acuity.

DME is the most common cause of vision impairment in the working-age diabetic population
and constitutes a significant public health burden given the global rise in diabetes
prevalence. The condition can affect individuals with both type 1 and type 2 diabetes,
with risk increasing with disease duration, poor glycaemic control, hypertension, and
nephropathy. Critically, the early stages of DME are frequently asymptomatic, meaning
that patients may experience subclinical structural damage before any functional deficit
is perceived.

OCT is the primary imaging modality for detecting and quantifying DME. The characteristic
findings include intraretinal cystoid spaces, subretinal fluid accumulation, and
generalised retinal thickening, all of which can be identified and measured with high
precision from cross-sectional OCT images. The ability to monitor retinal thickness
longitudinally makes OCT indispensable for evaluating treatment response following
anti-VEGF injections, laser photocoagulation, or corticosteroid therapy.

</div>
    """, unsafe_allow_html=True)

with dme_fact:
    st.warning("""
**Epidemiological context**

Estimated global prevalence: 21 million individuals with diabetes.

Leading cause of vision impairment in working-age adults with diabetes.

Prevalence is rising in parallel with the global diabetes epidemic.

Frequently asymptomatic in early stages, making screening particularly important.

Primary treatments: anti-VEGF injections, laser photocoagulation, corticosteroids.
    """)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# About the classifier
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='section-label'>System description</div>", unsafe_allow_html=True)
st.markdown(
    "<h2 style='font-size:1.35rem;margin-top:0'>About This Classifier</h2>",
    unsafe_allow_html=True,
)


st.markdown("""
<div class='card'>

This application is a research prototype implementing an automated classification
pipeline for retinal OCT images, assigning each scan to one of three diagnostic
categories: AMD, DME, or Normal.

Four deep learning architectures were developed and evaluated.

The first, referred to as CustomCNN, is a lightweight three-block convolutional
neural network trained from scratch on the OCT5k dataset. With approximately
93,440 trainable parameters, it serves as a computationally efficient baseline
and provides a reference point for assessing the gains attributable to transfer
learning.

The second architecture, EfficientNet-B0, leverages weights pre-trained on the
ImageNet dataset and was fine-tuned on the same OCT data using a two-stage
training procedure. The backbone was initially frozen to adapt the classification
head before being progressively unfrozen for end-to-end optimisation. The model
contains approximately 4 million parameters.

The third architecture, MIRAGE, is a Vision Transformer (ViT-Base) pre-trained
on multi-modal retinal imaging data including OCT, fundus photography, and
fluorescein angiography. It uses a patch size of 32×32 on 512×512 images,
producing a 16×16 grid of tokens processed by 12 transformer encoder blocks
with embedding dimension 768. It contains approximately 86 million parameters
and achieves a reported test balanced accuracy of 96.0% and AUC of 0.995 on OCT5k.

The fourth architecture, MedGemma, is based on the same SigLIP-So400m vision
encoder used inside Google's medical Vision-Language Model (medgemma-4b-it).
The encoder is loaded from the public google/siglip-so400m-patch14-384
checkpoint and kept entirely frozen during training. A lightweight MLP
classification head — comprising a Linear layer (1152 → 512), LayerNorm, ReLU
activation, Dropout, and a final Linear layer (512 → 3) — was trained on top
of the 1152-dimensional embeddings produced by the SigLIP encoder. This
linear-probe approach adapts general medical visual representations to the
specific OCT classification task without modifying the large pre-trained
backbone, and reaches a reported test accuracy of 90.0% on OCT5k.

All models were trained on the OCT5k dataset, comprising 3,231 images from
117 patients across the three diagnostic categories. A patient-level stratified
split was applied to prevent data leakage between training, validation, and
test partitions.

To ensure consistency between development and deployment conditions, all OCT
scans undergo an identical preprocessing pipeline during both training and
inference, including resizing and normalisation to 512×512 resolution.

The application additionally integrates an advanced uncertainty quantification
framework based on stochastic Monte Carlo (MC) Dropout inference. During
prediction, 20 stochastic forward passes are performed in order to estimate
predictive uncertainty and improve robustness to out-of-distribution (OOD)
inputs.

If the maximum class probability falls below the user-defined confidence
threshold, or if elevated Shannon entropy and predictive standard deviation
indicate substantial model uncertainty, the system automatically flags the
scan for human review. This mechanism is intended to reduce the risk of
automated misclassification in cases that fall outside the training
distribution, including severe imaging artefacts or retinal conditions not
represented in the dataset.

</div>
""", unsafe_allow_html=True)

st.info("""
### Two kinds of models in this app

CustomCNN and EfficientNet-B0 are trained specifically for this task. CustomCNN
learns everything from scratch using only the OCT5k images, while EfficientNet-B0
starts from general image features learned on ImageNet and is then fine-tuned on
OCT5k. Neither model has ever seen medical images before this project.

MIRAGE and MedGemma work differently. They are foundation models: very large
networks that were already pre-trained on huge amounts of data before this
project even started (MIRAGE on multi-modal retinal images, MedGemma on the
same vision encoder used inside Google's medical Vision-Language Model). For
this app, only a small classification head was trained on top of that existing
knowledge, rather than training the whole network from zero. This is generally
why they reach higher accuracy while needing less task-specific training data.
""")

st.info("""
### Trustworthiness & Safety Features

- Transparent preprocessing pipeline (512×512 normalisation for CNNs/MIRAGE; 384×384 RGB for MedGemma)
- Grad-CAM explainability visualisations (CustomCNN, EfficientNet)
- Attention Rollout saliency maps (MIRAGE, MedGemma)
- Monte Carlo Dropout uncertainty estimation
- Out-of-distribution detection
- Human oversight alerting mechanism
- Confidence-threshold-based safety gating
""")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown("""
| Architecture | Training strategy | Model type | Parameters | Test Accuracy |
|---|---|---|---|---|
| CustomCNN | From scratch | Standard | 93 k | 85.4% |
| EfficientNet-B0 | Transfer learning (ImageNet) | Standard | 4 M | 95.7% |
| MIRAGE | Pre-trained on retinal data | Foundation model | 86 M | 96.0% (BAcc) |
| MedGemma | Frozen SigLIP encoder + MLP head | Foundation model | ~3 B (frozen) + 0.6 M | 90.0% |
    """)
with col_t2:
    st.markdown("""
| Dataset | OCT5k |
|---|---|
| Total images | 3,231 |
| Patients | 117 |
| Split strategy | Patient-level stratified split |
| AMD images | 486 |
| DME images | 1,117 |
| Normal images | 1,628 |
    """)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Model performance on test set</div>", unsafe_allow_html=True)

perf_a, perf_b = st.columns(2, gap="large")

with perf_a:
    st.markdown("""
<div class='card' style='text-align:center'>
    <p style='font-family:Plus Jakarta Sans;font-size:0.8rem;font-weight:700;
    text-transform:uppercase;letter-spacing:0.06em;color:#6B7280;margin-bottom:6px'>
    CustomCNN — Baseline</p>
    <p style='font-family:Plus Jakarta Sans;font-size:3rem;font-weight:700;
    color:#2D3142;margin:0'>85.4%</p>
    <p style='font-size:0.82rem;color:#6B7280;margin-top:4px'>Test accuracy</p>
    <p style='font-size:0.83rem;color:#2D3142;margin-top:12px'>
    Lightweight 3-block CNN trained from scratch. ~93k parameters.
    Efficient on CPU, good baseline for comparison. Not a foundation model.</p>
</div>
""", unsafe_allow_html=True)

with perf_b:
    st.markdown("""
<div class='card' style='text-align:center;border-left:4px solid #84A59D'>
    <p style='font-family:Plus Jakarta Sans;font-size:0.8rem;font-weight:700;
    text-transform:uppercase;letter-spacing:0.06em;color:#84A59D;margin-bottom:6px'>
    EfficientNet-B0 — Recommended</p>
    <p style='font-family:Plus Jakarta Sans;font-size:3rem;font-weight:700;
    color:#84A59D;margin:0'>95.7%</p>
    <p style='font-size:0.82rem;color:#6B7280;margin-top:4px'>Test accuracy</p>
    <p style='font-size:0.83rem;color:#2D3142;margin-top:12px'>
    Transfer learning from ImageNet, two-stage fine-tuning. ~4M parameters.
    Higher accuracy and better uncertainty calibration. Not a foundation model.</p>
</div>
""", unsafe_allow_html=True)

perf_c, perf_d = st.columns(2, gap="large")

with perf_c:
    st.markdown("""
<div class='card' style='text-align:center;border-left:4px solid #7B9EA6'>
    <p style='font-family:Plus Jakarta Sans;font-size:0.8rem;font-weight:700;
    text-transform:uppercase;letter-spacing:0.06em;color:#7B9EA6;margin-bottom:6px'>
    MIRAGE — Vision Transformer</p>
    <p style='font-family:Plus Jakarta Sans;font-size:3rem;font-weight:700;
    color:#7B9EA6;margin:0'>96.0%</p>
    <p style='font-size:0.82rem;color:#6B7280;margin-top:4px'>Test balanced accuracy</p>
    <p style='font-size:0.83rem;color:#2D3142;margin-top:12px'>
    Foundation model: ViT-Base pre-trained on multi-modal retinal data. ~86M parameters.
    AUC 0.995 on OCT5k. Saliency via Attention Rollout.</p>
</div>
""", unsafe_allow_html=True)

with perf_d:
    st.markdown("""
<div class='card' style='text-align:center;border-left:4px solid #9B8EC4'>
    <p style='font-family:Plus Jakarta Sans;font-size:0.8rem;font-weight:700;
    text-transform:uppercase;letter-spacing:0.06em;color:#9B8EC4;margin-bottom:6px'>
    MedGemma — Medical VLM</p>
    <p style='font-family:Plus Jakarta Sans;font-size:3rem;font-weight:700;
    color:#9B8EC4;margin:0'>90.0%</p>
    <p style='font-size:0.82rem;color:#6B7280;margin-top:4px'>Test accuracy</p>
    <p style='font-size:0.83rem;color:#2D3142;margin-top:12px'>
    Foundation model: Google SigLIP vision encoder (frozen, ~3B params) + fine-tuned
    MLP head (~0.6M). Public checkpoint, no access token needed. Saliency via
    Attention Rollout on the SigLIP encoder.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Regulatory context
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='section-label'>Regulatory framework</div>", unsafe_allow_html=True)
st.markdown(
    "<h2 style='font-size:1.35rem;margin-top:0'>EU MDR and AI Act</h2>",
    unsafe_allow_html=True,
)

reg_a, reg_b = st.columns(2, gap="large")

with reg_a:
    st.markdown("""
<div class='card'>
<p class='section-label'>Medical Device Regulation 2017/745</p>

Under the EU Medical Device Regulation, software intended to provide information
used in making diagnostic or therapeutic decisions qualifies as a Software as Medical
Device (SaMD). Applying the classification rules set out in Annex VIII of the MDR,
and specifically Rule 11, a diagnostic decision-support tool of this type would be
classified as a Class IIa device, requiring conformity assessment by a Notified Body
prior to market placement.

This prototype has not been submitted for CE marking and is not intended for use in a
clinical setting. The classification analysis is provided for academic purposes to
contextualise the regulatory requirements that would apply in a real deployment scenario.
</div>
    """, unsafe_allow_html=True)

with reg_b:
    st.markdown("""
<div class='card'>
<p class='section-label'>EU AI Act — Regulation 2024/1689</p>

The EU AI Act classifies AI systems intended for use as or in medical devices as
high-risk systems under Annex III, point 5(b). High-risk AI systems are subject to
obligations including technical documentation, conformity assessment, post-market
monitoring, and specific requirements for transparency, human oversight, and robustness.    

This application was designed to reflect key trustworthiness principles
established under the EU AI Act within a research-oriented setting.

To support transparency obligations under Article 13, the system explicitly
discloses the preprocessing pipeline applied to all OCT scans during training
and inference.

To strengthen interpretability, Grad-CAM visual explanations are generated to
highlight the anatomical regions that most strongly contributed to each
prediction, enabling users to visually inspect model attention patterns.

Human oversight requirements under Article 14 are addressed through an
uncertainty-aware out-of-distribution detection mechanism based on Monte Carlo
Dropout inference. Predictions associated with elevated uncertainty or
insufficient confidence are automatically flagged for clinical review rather
than being treated as reliable autonomous outputs.                
</div>
    """, unsafe_allow_html=True)

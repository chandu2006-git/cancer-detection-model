import streamlit as st
import numpy as np
import json
import os
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermAI – Multimodal Skin Lesion Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load CSS ───────────────────────────────────────────────────────────────────
with open("style.css", "r", encoding="utf-8") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "AKIEC",
    "BCC",
    "BKL",
    "DF",
    "Melanoma",
    "Nevus",
    "VASC"
]
CLASS_MAPPING_PATH = "configs/class_mapping.json"
MODEL_PATH = "model/v4d_deployment.keras"
IMG_SIZE = (224, 224)

LOCALIZATION_OPTIONS = [
    "abdomen",
    "acral",
    "back",
    "chest",
    "ear",
    "face",
    "foot",
    "genital",
    "hand",
    "lower extremity",
    "neck",
    "scalp",
    "trunk",
    "unknown",
    "upper extremity"
]

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_class_mapping():
    if os.path.exists(CLASS_MAPPING_PATH):
        with open(CLASS_MAPPING_PATH) as f:
            return json.load(f)
    return {str(i): name for i, name in enumerate(CLASS_NAMES)}

# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess_image(image):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def encode_metadata(age, sex, localization):

    # 19 features
    meta = np.zeros((1, 19), dtype=np.float32)

    # Age
    meta[0, 0] = float(age)

    # Sex (3 columns)
    sex = sex.lower()

    if sex == "female":
        meta[0, 1] = 1

    elif sex == "male":
        meta[0, 2] = 1

    else:
        meta[0, 3] = 1

    # Localization (15 columns)
    localization = localization.lower()

    if localization in LOCALIZATION_OPTIONS:
        idx = LOCALIZATION_OPTIONS.index(localization)
        meta[0, 4 + idx] = 1
    else:
        meta[0, 4 + LOCALIZATION_OPTIONS.index("unknown")] = 1

    return meta
def run_inference(model, image, age, sex, localization):
    img_arr = preprocess_image(image)
    meta_arr = encode_metadata(age, sex, localization)

    try:
        return model.predict(
            [img_arr, meta_arr],
            verbose=0,
        )[0]

    except Exception as e:
        st.error(f"Inference error: {e}")
        return None

def donut_svg(pct):
    r, cx, cy, sw = 44, 60, 60, 12
    circ = 2 * 3.14159 * r
    dash = circ * pct / 100
    return f"""<svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#E8F0F8" stroke-width="{sw}"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#3B9EE8"
              stroke-width="{sw}" stroke-linecap="round"
              stroke-dasharray="{dash:.2f} {circ:.2f}"
              transform="rotate(-90 {cx} {cy})"/>
      <text x="{cx}" y="{cy+6}" text-anchor="middle"
            font-family="Inter,sans-serif" font-size="13" font-weight="700" fill="#1A2B4A">
        {pct:.2f}%
      </text>
    </svg>"""

def prob_bars_html(probs):
    rows = ""
    for cls, pct in probs.items():
        w = max(pct, 0.4)
        rows += f"""<div class="bar-row">
          <span class="bar-label">{cls}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{w:.2f}%"></div></div>
          <span class="bar-pct">{pct:.2f}%</span>
        </div>"""
    return f'<div class="bar-chart">{rows}</div>'

# ── Load resources ─────────────────────────────────────────────────────────────
model = load_model()
class_mapping = load_class_mapping()
model_status = "✅ Model loaded" if model else "⚠️ Demo mode (model not found)"

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-header">
  <div class="logo-area">
    <div class="logo-icon">🧠</div>
    <span class="logo-text">DermAI</span>
    <span class="logo-divider">|</span>
    <span class="logo-sub">Multimodal Skin Lesion Classification</span>
  </div>
</div>
<div class="header-divider"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TWO COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1, 1.55], gap="large")

# ── LEFT: Patient Information ──────────────────────────────────────────────────
with left_col:
    # Panel title
    st.markdown('<p class="panel-title">PATIENT INFORMATION</p>', unsafe_allow_html=True)

    # 1. Upload
    st.markdown('<p class="input-label">1. Upload Skin Lesion Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "img", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed", key="img_upload"
    )

    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, use_container_width=True)
        st.markdown(f'<div class="file-chip">🖼&nbsp;{uploaded_file.name}</div>',
                    unsafe_allow_html=True)
    else:
        pil_img = None
        st.markdown("""
        <div class="upload-placeholder">
          <div class="upload-icon">📂</div>
          <div class="upload-hint">Drag &amp; drop or click Browse files</div>
          <div class="upload-hint-sub">JPG · JPEG · PNG</div>
        </div>""", unsafe_allow_html=True)

    # 2. Age
    st.markdown('<p class="input-label">2. Age (years)</p>', unsafe_allow_html=True)
    age = st.slider("Age", 1, 100, 45, label_visibility="collapsed")

    # 3. Sex
    st.markdown('<p class="input-label">3. Sex</p>', unsafe_allow_html=True)
    sex = st.selectbox("Sex", ["Female", "Male"], label_visibility="collapsed")

    # 4. Localization
    st.markdown('<p class="input-label">4. Localization</p>', unsafe_allow_html=True)
    localization = st.selectbox("Localization", LOCALIZATION_OPTIONS,
                                label_visibility="collapsed")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button("✨  Analyze", use_container_width=True, type="primary")

# ── RIGHT: Prediction Result ───────────────────────────────────────────────────
with right_col:
    st.markdown('<p class="panel-title">PREDICTION RESULT</p>', unsafe_allow_html=True)

    if analyze_clicked:
        if not uploaded_file:
            st.warning("Please upload a skin lesion image first.")
        else:
            with st.spinner("Analyzing…"):
                if model:
                    preds = run_inference(model, pil_img, age, sex, localization)
                else:
                    # demo
                    raw = np.array([0.9182, 0.0427, 0.0211, 0.0102,
                                    0.0046, 0.0018, 0.0009, 0.0005])
                    preds = raw / raw.sum()

            if preds is not None:
                top_idx  = int(np.argmax(preds))
                top_cls  = class_mapping.get(str(top_idx), CLASS_NAMES[top_idx])
                top_pct  = float(preds[top_idx]) * 100
                conf_lbl = ("High Confidence" if top_pct >= 80
                            else "Medium Confidence" if top_pct >= 50
                            else "Low Confidence")
                badge_bg = "#FFE8E8" if top_pct >= 80 else "#FFF3CD"

                # ── Diagnosis card
                st.markdown(f"""
                <div class="result-card">
                  <div class="result-left">
                    <div class="result-sub">Most Likely Diagnosis</div>
                    <div class="result-diagnosis">{top_cls}</div>
                    <span class="conf-badge" style="background:{badge_bg}">{conf_lbl}</span>
                  </div>
                  <div class="result-mid">
                    <div class="result-sub">Confidence</div>
                    <div class="result-pct">{top_pct:.2f}%</div>
                  </div>
                  <div class="result-donut">{donut_svg(top_pct)}</div>
                </div>""", unsafe_allow_html=True)
                print("Predictions shape:", preds.shape)
                print("Number of predictions:", len(preds))
                print("Number of classes:", len(CLASS_NAMES))
                print(CLASS_NAMES)
                # ── Probability bars
                probs = {}
                assert len(CLASS_NAMES) == len(preds)
                probs = {
                    CLASS_NAMES[i]: float(preds[i]) * 100
                    for i in range(len(preds))
                }
                probs = dict(sorted(probs.items(), key=lambda x: -x[1]))

                st.markdown(f"""
                <div class="prob-card">
                  <p class="prob-title">Probability of All Classes</p>
                  {prob_bars_html(probs)}
                </div>""", unsafe_allow_html=True)

                # ── Research notes
                st.markdown("""
                <div class="notes-card">
                  <p class="notes-title">RESEARCH NOTES</p>
                  <div class="notes-grid">
                    <div>
                      <div class="note-item">🖥&nbsp; Model: DermAI V4D</div>
                      <div class="note-item">🔗&nbsp; Backbone: ViT-B16 (ImageNet-21k)</div>
                      <div class="note-item">📐&nbsp; Image Size: 224 × 224</div>
                    </div>
                    <div>
                      <div class="note-item">📊&nbsp; Metadata Used: Age, Sex, Localization</div>
                      <div class="note-item">✅&nbsp; Validation Accuracy: 83.04%</div>
                      <div class="note-item">🎯&nbsp; Melanoma Recall: 64.01%</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-placeholder">
          <div class="ph-icon">🔬</div>
          <div class="ph-text">Upload an image and fill in patient details,<br>
          then click <strong>Analyze</strong> to see results.</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
  <div class="footer-left">
    <span class="footer-shield">🛡</span>
    <div>
      <div class="footer-bold">Research Demonstration Only.</div>
      <div class="footer-muted">Not intended for medical diagnosis or clinical use.</div>
    </div>
  </div>
  <div class="footer-right">
    DermAI V4D &nbsp;•&nbsp; Vision Transformer &nbsp;•&nbsp; Multimodal AI
    &nbsp;&nbsp;<span class="model-tag">{model_status}</span>
  </div>
</div>""", unsafe_allow_html=True)
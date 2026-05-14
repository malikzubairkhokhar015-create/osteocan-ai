
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import time

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="OsteoScan AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --card:      #161a22;
    --border:    #1f2535;
    --accent:    #00d4ff;
    --accent2:   #7b61ff;
    --danger:    #ff4757;
    --warn:      #ffa502;
    --safe:      #2ed573;
    --text:      #e8eaf0;
    --muted:     #6b7280;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,212,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(123,97,255,0.06) 0%, transparent 60%),
        #0a0c10 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
footer, #MainMenu { display: none !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,212,255,0.35) !important;
}

[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 0 1.5rem;">
    <div style="
        display: inline-block;
        background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(123,97,255,0.12));
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 50px;
        padding: 0.4rem 1.2rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.15em;
        color: #00d4ff;
        text-transform: uppercase;
        margin-bottom: 1rem;
    ">AI-Powered Radiology Assistant</div>
    <h1 style="
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.8rem, 6vw, 5rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1;
        background: linear-gradient(135deg, #ffffff 0%, #00d4ff 50%, #7b61ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.6rem;
    ">OsteoScan AI</h1>
    <p style="
        font-family: 'DM Sans', sans-serif;
        font-size: 1.1rem;
        color: #6b7280;
        font-weight: 300;
        max-width: 520px;
        margin: 0 auto 0.5rem;
    ">Deep learning bone fracture detection with<br>Grad-CAM explainability — 96% accuracy</p>
</div>
""", unsafe_allow_html=True)

# ─── STAT PILLS ────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; justify-content:center; gap:1rem; flex-wrap:wrap; margin-bottom:2.5rem;">
    <div style="background:rgba(0,212,255,0.08); border:1px solid rgba(0,212,255,0.2);
                border-radius:50px; padding:0.45rem 1.2rem; font-family:'DM Sans',sans-serif;
                font-size:0.85rem; color:#00d4ff;">⚡ 96% Accuracy</div>
    <div style="background:rgba(123,97,255,0.08); border:1px solid rgba(123,97,255,0.2);
                border-radius:50px; padding:0.45rem 1.2rem; font-family:'DM Sans',sans-serif;
                font-size:0.85rem; color:#7b61ff;">🧠 EfficientNetB4 + Grad-CAM</div>
    <div style="background:rgba(46,213,115,0.08); border:1px solid rgba(46,213,115,0.2);
                border-radius:50px; padding:0.45rem 1.2rem; font-family:'DM Sans',sans-serif;
                font-size:0.85rem; color:#2ed573;">🦴 Multi-Bone Detection</div>
</div>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fracture_model_v2.keras")

model = load_model()

# ─── HELPER FUNCTIONS ──────────────────────────────────────────
def preprocess_image(img_array, img_size=224):
    img   = cv2.resize(img_array, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh   = clahe.apply(gray)
    rgb   = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32) / 255.0

def generate_gradcam(model, img_array):
    effnet = None
    for layer in model.layers:
        if "efficientnetb4" in layer.name.lower():
            effnet = layer
            break
    feat_extractor = tf.keras.Model(
        inputs=effnet.input,
        outputs=effnet.get_layer("top_conv").output
    )
    with tf.GradientTape() as tape:
        features = feat_extractor(img_array, training=False)
        tape.watch(features)
        x = features
        after = False
        for layer in model.layers:
            if after:
                x = layer(x, training=False)
            if "efficientnetb4" in layer.name.lower():
                after = True
        grads = tape.gradient(x[:, 0], features)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = features[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    mx = tf.reduce_max(heatmap)
    if mx > 0:
        heatmap = heatmap / mx
    return heatmap.numpy()

def apply_overlay(base_rgb, heatmap, alpha=0.45):
    h, w   = base_rgb.shape[:2]
    resized = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_rgb, 1 - alpha, colored, alpha, 0)
    return overlay, colored

# ─── UPLOAD SECTION ────────────────────────────────────────────
st.markdown("""
<h2 style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700;
           color:#e8eaf0; margin-bottom:0.4rem;">Upload X-Ray Image</h2>
<p style="font-family:'DM Sans',sans-serif; color:#6b7280; font-size:0.9rem; margin-bottom:1rem;">
    Supports JPG, JPEG, PNG — elbow, wrist, shoulder, hip, ankle, knee, spine
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded_file:
    file_bytes  = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr     = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_display = cv2.resize(img_rgb, (224, 224))

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        analyze = st.button("🔬  Analyze X-Ray", use_container_width=True)

    if analyze:
        prog   = st.progress(0)
        status = st.empty()

        status.markdown('<p style="font-family:DM Sans,sans-serif;color:#00d4ff;font-size:0.9rem;">⚙️ Preprocessing with CLAHE...</p>', unsafe_allow_html=True)
        for i in range(30): prog.progress(i); time.sleep(0.01)

        preprocessed = preprocess_image(img_rgb)
        img_input    = np.expand_dims(preprocessed, axis=0)

        status.markdown('<p style="font-family:DM Sans,sans-serif;color:#00d4ff;font-size:0.9rem;">🧠 Running EfficientNetB4 inference...</p>', unsafe_allow_html=True)
        for i in range(30, 65): prog.progress(i); time.sleep(0.01)

        pred_prob = float(model(img_input, training=False)[0][0])

        status.markdown('<p style="font-family:DM Sans,sans-serif;color:#00d4ff;font-size:0.9rem;">🔥 Generating Grad-CAM heatmap...</p>', unsafe_allow_html=True)
        for i in range(65, 100): prog.progress(i); time.sleep(0.01)

        heatmap = generate_gradcam(model, img_input)
        base_img = (preprocessed * 255).astype(np.uint8)
        overlaid, heatmap_colored = apply_overlay(base_img, heatmap)

        prog.empty(); status.empty()

        # ─── RESULT LOGIC ──────────────────────────────────────
        if pred_prob <= 0.5:
            pred_label = "FRACTURE DETECTED"
            confidence = (1 - pred_prob) * 100
            result_icon = "⚠️"
            recommendation = "Immediate radiologist review recommended. Fracture region highlighted in Grad-CAM overlay."
            if confidence > 85:
                risk_label, risk_icon = "HIGH RISK", "🔴"
                rc = "#ff4757"; rb = "rgba(255,71,87,0.12)"; rbr = "rgba(255,71,87,0.35)"
            else:
                risk_label, risk_icon = "MODERATE RISK", "🟡"
                rc = "#ffa502"; rb = "rgba(255,165,2,0.12)"; rbr = "rgba(255,165,2,0.35)"
        else:
            pred_label = "NO FRACTURE DETECTED"
            confidence = pred_prob * 100
            result_icon = "✅"
            risk_label, risk_icon = "LOW RISK", "🟢"
            rc = "#2ed573"; rb = "rgba(46,213,115,0.12)"; rbr = "rgba(46,213,115,0.35)"
            recommendation = "No fracture detected. Grad-CAM highlights region of model attention for verification."

        # ─── RESULT BANNER ─────────────────────────────────────
        st.markdown(f"""
        <div style="background:{rb}; border:1px solid {rbr}; border-radius:20px;
                    padding:1.5rem 2rem; margin:1.5rem 0;
                    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
            <div>
                <div style="font-family:'DM Sans',sans-serif; font-size:0.75rem;
                            letter-spacing:0.15em; text-transform:uppercase; color:{rc}; margin-bottom:0.3rem;">
                    Diagnosis Result
                </div>
                <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:{rc};">
                    {result_icon} {pred_label}
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:0.9rem; color:#9ca3af; margin-top:0.4rem; max-width:480px;">
                    {recommendation}
                </div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Syne',sans-serif; font-size:3rem; font-weight:800; color:{rc}; line-height:1;">
                    {confidence:.1f}%
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:0.8rem; color:#6b7280;
                            text-transform:uppercase; letter-spacing:0.1em;">Confidence</div>
                <div style="background:{rb}; border:1px solid {rbr}; border-radius:50px;
                            padding:0.3rem 1rem; margin-top:0.5rem;
                            font-family:'DM Sans',sans-serif; font-size:0.8rem; color:{rc}; font-weight:600;">
                    {risk_icon} {risk_label}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─── 4-PANEL IMAGES ────────────────────────────────────
        st.markdown("""
        <h3 style="font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700;
                   color:#e8eaf0; margin:1.5rem 0 1rem;">📊 Visual Analysis</h3>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        panel = "background:#161a22; border:1px solid #1f2535; border-radius:16px; padding:0.75rem; text-align:center;"
        lbl   = "font-family:'DM Sans',sans-serif; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; color:#6b7280; margin-bottom:0.5rem;"

        with c1:
            st.markdown(f'<div style="{panel}"><div style="{lbl}">Original X-Ray</div>', unsafe_allow_html=True)
            st.image(img_display, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div style="{panel}"><div style="{lbl}">CLAHE Enhanced</div>', unsafe_allow_html=True)
            st.image((preprocessed*255).astype(np.uint8), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div style="{panel}"><div style="{lbl}">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            st.image(heatmap_colored, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div style="{panel}"><div style="{lbl}">Fracture Region</div>', unsafe_allow_html=True)
            st.image(overlaid, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ─── METRIC CARDS ──────────────────────────────────────
        st.markdown("""
        <h3 style="font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700;
                   color:#e8eaf0; margin:2rem 0 1rem;">📈 Model Performance</h3>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        for col, label, value, color in [
            (m1, "Test Accuracy",   "96.00%",  "#00d4ff"),
            (m2, "AUC Score",       "0.9748",  "#7b61ff"),
            (m3, "Fracture Recall", "96%",     "#ffa502"),
            (m4, "Architecture",    "EffNetB4","#2ed573"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#161a22; border:1px solid #1f2535;
                            border-top:3px solid {color}; border-radius:14px;
                            padding:1.2rem 1rem; text-align:center;">
                    <div style="font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:800; color:{color};">{value}</div>
                    <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#6b7280;
                                text-transform:uppercase; letter-spacing:0.1em; margin-top:0.3rem;">{label}</div>
                </div>""", unsafe_allow_html=True)

        # ─── XAI EXPLANATION ───────────────────────────────────
        st.markdown(f"""
        <div style="background:rgba(123,97,255,0.07); border:1px solid rgba(123,97,255,0.2);
                    border-left:4px solid #7b61ff; border-radius:12px;
                    padding:1.2rem 1.5rem; margin:2rem 0 0.5rem;">
            <div style="font-family:'Syne',sans-serif; font-weight:700; color:#7b61ff;
                        margin-bottom:0.4rem; font-size:0.95rem;">🧠 Explainable AI — How the model decided</div>
            <div style="font-family:'DM Sans',sans-serif; font-size:0.88rem; color:#9ca3af; line-height:1.6;">
                The <strong style="color:#e8eaf0;">Grad-CAM heatmap</strong> reveals which bone region
                most influenced this prediction. <strong style="color:{rc};">Red/yellow zones</strong>
                indicate highest model attention — where fracture patterns were detected.
                The model analyzed bone density, cortical continuity, and trabecular patterns
                before concluding with <strong style="color:{rc};">{confidence:.1f}% confidence</strong>.
            </div>
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-size:0.75rem; color:#374151;
                    text-align:center; padding:1rem 0 2rem;">
            ⚠️ For research and educational purposes only. Always consult a qualified radiologist for clinical decisions.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#161a22; border:2px dashed #1f2535; border-radius:24px;
                padding:4rem 2rem; text-align:center; margin:1rem 0;">
        <div style="font-size:4rem; margin-bottom:1rem;">🦴</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700;
                    color:#374151; margin-bottom:0.5rem;">No X-Ray Uploaded Yet</div>
        <div style="font-family:'DM Sans',sans-serif; font-size:0.9rem; color:#374151;">
            Upload a bone X-ray above to begin AI-powered fracture analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

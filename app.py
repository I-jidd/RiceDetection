import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import pickle
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis
from datetime import datetime
import requests
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
import io

# ──────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Leaf Disease Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",   # always try to expand
)

# ──────────────────────────────────────────────
# GLOBAL CSS — dark theme + glassmorphism
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Root / App shell ── */
html, body, .stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #0d1f14 100%) !important;
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
footer { visibility: hidden; }
button[data-testid="collapsedControl"] { background: #22c55e !important; border-radius: 50% !important; box-shadow: 0 0 14px rgba(34,197,94,0.7) !important; }
.block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

/* ── Animated gradient title ── */
.main-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    background: linear-gradient(90deg, #22c55e, #86efac, #4ade80, #22c55e);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    animation: gradientShift 4s ease infinite;
    margin-bottom: 0.4rem;
    line-height: 1.2;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.sub-header {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: transform .25s ease, box-shadow .25s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(34,197,94,0.15);
}

/* ── Result (green glow) card ── */
.result-card {
    background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(134,239,172,0.05) 100%);
    border: 1px solid rgba(134,239,172,0.35);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 0 30px rgba(34,197,94,0.12);
}

/* ── Stat pills ── */
.stat-pill {
    display: inline-block;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(134,239,172,0.3);
    border-radius: 50px;
    padding: 6px 18px;
    font-size: 0.85rem;
    color: #86efac;
    font-weight: 600;
    margin: 4px;
}

/* ── Feature tag ── */
.feature-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #a5b4fc;
    margin: 3px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.03em !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(34,197,94,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 25px rgba(34,197,94,0.45) !important;
    filter: brightness(1.1) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Secondary / danger buttons ── */
[data-testid="stButton-secondary"] > button {
    background: rgba(239,68,68,0.15) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    box-shadow: none !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #94a3b8 !important;
    font-weight: 500;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: #fff !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #22c55e, #86efac) !important;
    border-radius: 10px !important;
}
.stProgress > div > div > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0f1a0f 100%) !important;
    border-right: 1px solid rgba(34,197,94,0.15) !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(34,197,94,0.4);
    border-radius: 16px;
    padding: 10px;
    transition: border-color .3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #22c55e;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Info / success / warning boxes ── */
.stAlert {
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #86efac !important; font-weight: 700 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(34,197,94,0.4); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #22c55e; }

/* ── History cards ── */
.history-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: background .2s;
}
.history-card:hover { background: rgba(34,197,94,0.06); }

/* ── Disease badge ── */
.disease-badge {
    display: inline-block;
    background: linear-gradient(135deg, #22c55e20, #86efac10);
    border: 1px solid #86efac55;
    border-radius: 50px;
    padding: 4px 16px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #86efac;
}

/* ── About feature blocks ── */
.about-feature {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    height: 100%;
}
.about-feature h4 { color: #a5b4fc; margin-bottom: 8px; font-size: 1rem; }
.about-feature p  { color: #94a3b8; font-size: 0.82rem; line-height: 1.5; }

/* ── Animated pulse for prediction result ── */
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 20px rgba(34,197,94,0.2); }
    50%       { box-shadow: 0 0 40px rgba(34,197,94,0.5); }
}
.result-card.active { animation: pulseGreen 2.5s ease infinite; }

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: #22c55e !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# LOTTIE LOADER
# ══════════════════════════════════════════════
def load_lottie(url: str):
    """Fetch Lottie JSON with graceful fallback."""
    try:
        r = requests.get(url, timeout=7)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# Preload animation URLs (all free LottieFiles CDN)
LOTTIE_HOME    = "https://assets2.lottiefiles.com/packages/lf20_jtbfg2nb.json"  # plant / leaf
LOTTIE_LOADING = "https://assets4.lottiefiles.com/packages/lf20_qp1q7mct.json"  # scanning
LOTTIE_EMPTY   = "https://assets3.lottiefiles.com/packages/lf20_ysrn2iwp.json"  # empty state
LOTTIE_SUCCESS = "https://assets9.lottiefiles.com/packages/lf20_lk80fpsm.json"  # checkmark


# ══════════════════════════════════════════════
# FEATURE EXTRACTION — HELPER UTILS
# ══════════════════════════════════════════════
def safe_stat(func, arr, default=0.0):
    arr = np.asarray(arr).astype(np.float32).ravel()
    if arr.size == 0:
        return float(default)
    try:
        val = func(arr)
        return float(default) if (np.isnan(val) or np.isinf(val)) else float(val)
    except Exception:
        return float(default)


def create_green_leaf_mask(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv,
                       np.array([25, 20, 20], np.uint8),
                       np.array([100, 255, 255], np.uint8))
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == idx, 255, 0).astype(np.uint8)
    return mask


def create_lesion_mask(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    cond1 = (h >= 5) & (h <= 40) & (s >= 40) & (v >= 40)
    cond2 = (b >= 140)
    cond3 = (a >= 128)
    lesion = np.where((cond1 & cond2) | (cond1 & cond3), 255, 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, k)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, k)
    return lesion


def extract_color_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    for i, n in enumerate(['r', 'g', 'b']):
        ch = img_rgb[:, :, i]
        feats[f'rgb_mean_{n}'] = float(np.mean(ch))
        feats[f'rgb_std_{n}']  = float(np.std(ch))
        feats[f'rgb_skew_{n}'] = safe_stat(skew, ch)
    for i, n in enumerate(['h', 's', 'v']):
        ch = img_hsv[:, :, i]
        feats[f'hsv_mean_{n}'] = float(np.mean(ch))
        feats[f'hsv_std_{n}']  = float(np.std(ch))
    for i, n in enumerate(['l', 'a', 'b']):
        ch = img_lab[:, :, i]
        feats[f'lab_mean_{n}'] = float(np.mean(ch))
        feats[f'lab_std_{n}']  = float(np.std(ch))
    feats['gray_mean']     = float(np.mean(img_gray))
    feats['gray_std']      = float(np.std(img_gray))
    feats['gray_skew']     = safe_stat(skew, img_gray)
    feats['gray_kurtosis'] = safe_stat(kurtosis, img_gray)
    hist_d, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
    feats['gray_entropy']  = float(-np.sum(hist_d * np.log2(hist_d + 1e-12)))
    for i, n in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([img_rgb], [i], None, [8], [0, 256]).flatten()
        hist /= (hist.sum() + 1e-12)
        for j, val in enumerate(hist):
            feats[f'rgb_hist_{n}_{j}'] = float(val)
    return feats


def extract_glcm_features(img_gray):
    feats = {}
    glcm = graycomatrix(img_gray, distances=[1, 2],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        vals = graycoprops(glcm, prop).flatten()
        feats[f'glcm_{prop}_mean'] = float(np.mean(vals))
        feats[f'glcm_{prop}_std']  = float(np.std(vals))
    return feats


def extract_lbp_features(img_gray, radius=1, n_points=8):
    feats = {}
    lbp    = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    # uniform LBP has n_points+2 possible values (0..n_points+1)
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    for i, val in enumerate(hist):
        feats[f'lbp_hist_{i}'] = float(val)
    feats['lbp_mean'] = float(np.mean(lbp))
    feats['lbp_std']  = float(np.std(lbp))
    return feats


def extract_shape_features(img_rgb):
    feats = {}
    mask  = create_green_leaf_mask(img_rgb)
    props = regionprops(label(mask > 0))
    defaults = ['leaf_area','leaf_perimeter','leaf_bbox_w','leaf_bbox_h',
                'leaf_aspect_ratio','leaf_extent','leaf_solidity',
                'leaf_equiv_diameter','leaf_eccentricity']
    if not props:
        return {k: 0.0 for k in defaults}
    r = max(props, key=lambda x: x.area)
    minr, minc, maxr, maxc = r.bbox
    bh, bw = maxr - minr, maxc - minc
    feats['leaf_area']           = float(r.area)
    feats['leaf_perimeter']      = float(r.perimeter)
    feats['leaf_bbox_w']         = float(bw)
    feats['leaf_bbox_h']         = float(bh)
    feats['leaf_aspect_ratio']   = float(bw / (bh + 1e-12))
    feats['leaf_extent']         = float(r.extent)
    feats['leaf_solidity']       = float(r.solidity)
    feats['leaf_equiv_diameter'] = float(r.equivalent_diameter_area)
    feats['leaf_eccentricity']   = float(r.eccentricity)
    return feats


def extract_hog_features(img_gray):
    feats = {}
    hog_vec = hog(img_gray, orientations=9, pixels_per_cell=(32, 32),
                  cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    for i, val in enumerate(hog_vec):
        feats[f'hog_{i}'] = float(val)
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(gx**2 + gy**2)
    feats['grad_mean'] = float(np.mean(gm))
    feats['grad_std']  = float(np.std(gm))
    return feats


def extract_lesion_features(img_rgb):
    feats = {}
    lesion_mask = create_lesion_mask(img_rgb)
    leaf_mask   = create_green_leaf_mask(img_rgb)
    lesion_mask = cv2.bitwise_and(lesion_mask, lesion_mask, mask=leaf_mask)
    props       = regionprops(label(lesion_mask > 0))
    leaf_area   = np.sum(leaf_mask > 0)
    lesion_area = np.sum(lesion_mask > 0)
    feats['lesion_area']         = float(lesion_area)
    feats['lesion_ratio']        = float(lesion_area / (leaf_area + 1e-12))
    feats['lesion_count']        = float(len(props))
    if not props:
        feats['lesion_mean_area']      = 0.0
        feats['lesion_largest_area']   = 0.0
        feats['lesion_perimeter_sum']  = 0.0
    else:
        areas = [p.area for p in props]
        feats['lesion_mean_area']     = float(np.mean(areas))
        feats['lesion_largest_area']  = float(np.max(areas))
        feats['lesion_perimeter_sum'] = float(sum(p.perimeter for p in props))
    return feats


def extract_all_features(img_rgb, img_gray, img_hsv, img_lab):
    """Combine all feature extractors into one dict."""
    feats = {}
    feats.update(extract_color_features(img_rgb, img_gray, img_hsv, img_lab))
    feats.update(extract_glcm_features(img_gray))
    feats.update(extract_lbp_features(img_gray))
    feats.update(extract_shape_features(img_rgb))
    feats.update(extract_hog_features(img_gray))
    feats.update(extract_lesion_features(img_rgb))
    return feats


# ══════════════════════════════════════════════
# MODEL LOADING  (cached)
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner="🌾 Loading model artifacts…")
def load_artifacts():
    model  = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    imputer= joblib.load('imputer.pkl')
    le     = joblib.load('label_encoder.pkl')
    with open('selected_features.pkl', 'rb') as f:
        selected_cols = pickle.load(f)
    with open('all_feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, imputer, le, selected_cols, feature_cols

model, scaler, imputer, le, selected_cols, feature_cols = load_artifacts()


# ══════════════════════════════════════════════
# IMAGE PREPROCESSING  (unified pipeline)
# ══════════════════════════════════════════════
def preprocess_image(file_source):
    """
    Accept uploaded file OR camera bytes.
    Returns (img_rgb, img_gray, img_hsv, img_lab) all at 256×256.
    """
    file_bytes = np.asarray(bytearray(file_source.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, None, None, None
    img_bgr  = cv2.resize(img_bgr, (256, 256))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return img_rgb, img_gray, img_hsv, img_lab


# ══════════════════════════════════════════════
# PREDICTION PIPELINE
# ══════════════════════════════════════════════
def predict_image(img_rgb, img_gray, img_hsv, img_lab):
    """Return dict with class_name, confidence, top-3 classes+probs."""
    feats_dict = extract_all_features(img_rgb, img_gray, img_hsv, img_lab)
    df         = pd.DataFrame([feats_dict]).reindex(columns=feature_cols)
    X_raw      = df.values.astype(float)
    X_imp      = imputer.transform(X_raw)
    X_sc       = scaler.transform(X_imp)
    sel_idx    = [feature_cols.index(c) for c in selected_cols]
    X_sel      = X_sc[:, sel_idx]
    pred_label = model.predict(X_sel)[0]
    class_name = le.inverse_transform([pred_label])[0]
    proba      = (model.predict_proba(X_sel)[0]
                  if hasattr(model, "predict_proba")
                  else np.array([1.0]))
    class_labels = le.inverse_transform(range(len(proba)))
    top3  = sorted(zip(class_labels, proba), key=lambda x: x[1], reverse=True)[:3]
    return {
        "class_name":   class_name,
        "confidence":   float(max(proba)),
        "probabilities": proba,
        "class_labels":  class_labels,
        "top3":          top3,
        "img_rgb":       img_rgb.copy(),
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ══════════════════════════════════════════════
# DOWNLOAD HELPER — annotated PNG
# ══════════════════════════════════════════════
def build_download_image(pred: dict) -> bytes:
    """Return PNG bytes of the image annotated with prediction."""
    pil_img = Image.fromarray(pred["img_rgb"]).resize((512, 512))
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    # Semi-transparent banner
    draw.rectangle([(0, 0), (512, 80)], fill=(15, 23, 42, 200))
    try:
        font_big   = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font_big = font_small = ImageFont.load_default()
    draw.text((16, 10), pred["class_name"],
              fill=(134, 239, 172), font=font_big)
    draw.text((16, 48), f"Confidence: {pred['confidence']*100:.1f}%  |  {pred['timestamp']}",
              fill=(148, 163, 184), font=font_small)
    pil_img = pil_img.convert("RGBA")
    pil_img.alpha_composite(overlay)
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════
defaults = {"history": [], "current_prediction": None, "active_page": "🏠 Home"}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v



# ══════════════════════════════════════════════
# TOP NAVIGATION BAR (always visible fallback)
# ══════════════════════════════════════════════
st.markdown("""
<style>
.topnav {
    display: flex;
    justify-content: center;
    gap: 10px;
    padding: 10px 0 18px 0;
    flex-wrap: wrap;
}
.topnav a {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    color: #cbd5e1 !important;
    border-radius: 50px;
    padding: 8px 22px;
    font-size: 0.88rem;
    font-weight: 500;
    text-decoration: none !important;
    transition: all .2s;
    cursor: pointer;
}
.topnav a:hover {
    background: rgba(34,197,94,0.18);
    border-color: #22c55e;
    color: #86efac !important;
}
.topnav a.active {
    background: linear-gradient(135deg,#22c55e,#16a34a);
    border-color: transparent;
    color: #fff !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

_pages = ["🏠 Home", "🔬 Classifier", "📜 History", "ℹ️ About"]
_cols  = st.columns(len(_pages))
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠 Home"

for _col, _page in zip(_cols, _pages):
    with _col:
        _active = "primary" if st.session_state.nav_page == _page else "secondary"
        if st.button(_page, key=f"topnav_{_page}", width="stretch",
                     type="primary" if st.session_state.nav_page == _page else "secondary"):
            st.session_state.nav_page = _page
            st.rerun()

st.markdown("<hr style='border-color:rgba(255,255,255,0.07); margin:0 0 18px 0;'>",
            unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════
with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div style="text-align:center; padding:12px 0 4px 0;">
        <div style="font-size:2.8rem; line-height:1;">🌾</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem;
                    font-weight:700; color:#86efac; letter-spacing:0.04em;">
            Rice AI Diagnostics
        </div>
        <div style="font-size:0.72rem; color:#475569; margin-top:2px;">
            v2.0 · Powered by ML
        </div>
    </div>
    <hr style="margin:12px 0; border-color:rgba(255,255,255,0.08);">
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["🏠 Home", "🔬 Classifier", "📜 History", "ℹ️ About"],
        icons=["house-fill", "camera2", "clock-history", "info-circle-fill"],
        default_index=0,
        styles={
            "container":         {"padding": "0", "background": "transparent"},
            "icon":              {"color": "#86efac", "font-size": "17px"},
            "nav-link":          {
                "font-size": "14px", "text-align": "left",
                "margin": "4px 0", "border-radius": "12px",
                "color": "#cbd5e1", "padding": "10px 16px",
                "transition": "all .2s",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg,#22c55e,#16a34a)",
                "color": "#ffffff", "font-weight": "600",
            },
        },
    )
    st.session_state.active_page = selected
    # Keep top-nav in sync with sidebar
    if selected != st.session_state.get("nav_page"):
        st.session_state.nav_page = selected

    # Sidebar stats strip
    st.markdown("<hr style='margin:16px 0; border-color:rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
    n_hist = len(st.session_state.history)
    c1, c2 = st.columns(2)
    c1.metric("Analyses", n_hist)
    classes = list(set(h["disease"] for h in st.session_state.history)) if st.session_state.history else []
    c2.metric("Diseases", len(classes))

    if n_hist > 0:
        avg_conf = np.mean([h["confidence"] for h in st.session_state.history])
        st.progress(avg_conf / 100, text=f"Avg confidence: {avg_conf:.1f}%")

    st.markdown("""
    <div style="margin-top:auto; text-align:center; color:#334155;
                font-size:0.68rem; padding:12px 0 4px 0; letter-spacing:0.04em;">
        🌾 Rice AI Diagnostics © 2025<br>OpenCV · scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ①  HOME PAGE
# ══════════════════════════════════════════════
selected = st.session_state.get("nav_page", "🏠 Home")
if selected == "🏠 Home":
    st.markdown('<h1 class="main-header">🌾 Rice Leaf Disease Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Precision diagnostics powered by handcrafted ML features</p>',
                unsafe_allow_html=True)

    col_anim, col_info = st.columns([1, 1], gap="large")

    with col_anim:
        lottie_home = load_lottie(LOTTIE_HOME)
        if lottie_home:
            st_lottie(lottie_home, speed=0.9, height=380, key="lottie_home")
        else:
            st.markdown("""
            <div style="height:380px; display:flex; align-items:center; justify-content:center;
                        background:rgba(34,197,94,0.06); border:1px solid rgba(34,197,94,0.2);
                        border-radius:20px; font-size:5rem; text-align:center;">
                🌿
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#86efac; margin-top:0; font-family:'Space Grotesk',sans-serif;">
                What can it diagnose?
            </h3>
            <p style="color:#94a3b8; line-height:1.7; font-size:0.92rem;">
                Instantly identifies common rice leaf diseases — including
                <strong style="color:#e2e8f0;">Blast, Brown Spot, Bacterial Blight,</strong>
                and <strong style="color:#e2e8f0;">Healthy</strong> leaves — using over
                100 handcrafted image features.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature modules pills
        modules = ["🎨 Color Analysis", "🔳 GLCM Texture", "🔁 LBP Patterns",
                   "📐 Leaf Shape", "🌀 HOG Gradients", "🔴 Lesion Detection"]
        pills_html = " ".join(f'<span class="stat-pill">{m}</span>' for m in modules)
        st.markdown(f'<div class="glass-card">{pills_html}</div>', unsafe_allow_html=True)

        # CTA button → navigate to Classifier
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Start Classifying", width="stretch"):
            # Streamlit can't directly mutate option_menu, but we can signal intent
            st.info("👈 Click **🔬 Classifier** in the sidebar to upload your leaf image!", icon="💡")

    # ── Stats row ──
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Feature Types",    "6")
    m2.metric("Total Features",   "100+")
    m3.metric("Supported Formats","JPG / PNG")
    m4.metric("Webcam Support",   "✅ Yes")

    # ── How it works ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🗺️ How It Works")
    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("📤", "Upload or Capture", "Choose an image or use your webcam to capture a rice leaf."),
        ("⚙️", "Feature Extraction", "Color, texture, shape & lesion features are computed in milliseconds."),
        ("🤖", "ML Prediction",      "Preprocessed features are fed to the trained classifier."),
        ("📊", "Results & History",  "View prediction, confidence chart, and download annotated image."),
    ]
    for col, (icon, title, desc) in zip([s1, s2, s3, s4], steps):
        col.markdown(f"""
        <div class="about-feature">
            <div style="font-size:2rem; margin-bottom:8px;">{icon}</div>
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ②  CLASSIFIER PAGE
# ══════════════════════════════════════════════
elif selected == "🔬 Classifier":
    st.markdown('<h1 class="main-header" style="font-size:2.4rem;">🔬 Classify Rice Leaf</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="margin-bottom:1.2rem;">Upload or capture an image to diagnose</p>',
                unsafe_allow_html=True)

    left_col, right_col = st.columns([1.05, 1], gap="large")

    # ─────────── LEFT: input ───────────
    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Image Input")
        tab_up, tab_cam = st.tabs(["📁 Upload Image", "📷 Webcam Capture"])

        uploaded_file = None
        camera_file   = None

        with tab_up:
            uploaded_file = st.file_uploader(
                "Drag & drop or click to browse",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
        with tab_cam:
            st.caption("Make sure your camera is accessible in the browser.")
            camera_file = st.camera_input("Take a photo", label_visibility="collapsed")

        active_file = uploaded_file if uploaded_file is not None else camera_file

        if active_file is not None:
            # Decode & show preview
            img_rgb, img_gray, img_hsv, img_lab = preprocess_image(active_file)

            if img_rgb is None:
                st.error("❌ Could not decode image. Please try a valid JPG/PNG file.")
            else:
                st.image(img_rgb, caption="Preview — 256 × 256 px", width="stretch")

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    run_predict = st.button("🔍 Predict Disease", type="primary", width="stretch")
                with btn_col2:
                    if st.button("🧹 Clear", width="stretch"):
                        st.session_state.current_prediction = None
                        st.rerun()

                if run_predict:
                    with st.spinner("🔬 Extracting features & predicting…"):
                        lottie_scan = load_lottie(LOTTIE_LOADING)
                        anim_slot   = st.empty()
                        if lottie_scan:
                            with anim_slot:
                                st_lottie(lottie_scan, height=120, key="lottie_scan")

                        pred = predict_image(img_rgb, img_gray, img_hsv, img_lab)
                        anim_slot.empty()

                    st.session_state.current_prediction = pred

                    # Append to history (avoid duplicates in same second)
                    last_ts = st.session_state.history[-1]["timestamp"] if st.session_state.history else ""
                    if pred["timestamp"] != last_ts:
                        st.session_state.history.append({
                            "timestamp":  pred["timestamp"],
                            "disease":    pred["class_name"],
                            "confidence": round(pred["confidence"] * 100, 1),
                            "top3":       pred["top3"],
                        })

                    st.toast(f"✅ Predicted: {pred['class_name']} ({pred['confidence']*100:.1f}%)",
                             icon="🌾")
        else:
            # Idle state placeholder
            st.markdown("""
            <div style="height:220px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center;
                        border:2px dashed rgba(34,197,94,0.25); border-radius:16px;
                        color:#4b5563; font-size:0.9rem; gap:8px;">
                <span style="font-size:2.5rem;">🌿</span>
                <span>No image selected</span>
                <span style="font-size:0.78rem;">Upload an image or use your webcam</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)   # close glass-card

    # ─────────── RIGHT: results ───────────
    with right_col:
        st.markdown('<div class="glass-card" style="min-height:520px;">', unsafe_allow_html=True)
        st.markdown("#### 📊 Prediction Results")

        if st.session_state.current_prediction:
            pred = st.session_state.current_prediction

            # Disease badge + confidence
            conf_pct = pred["confidence"] * 100
            badge_color = "#22c55e" if conf_pct >= 70 else "#f59e0b" if conf_pct >= 45 else "#ef4444"

            st.markdown(f"""
            <div class="result-card active">
                <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;
                            letter-spacing:.08em; margin-bottom:6px;">Diagnosed Disease</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.9rem;
                            font-weight:700; color:#86efac; margin-bottom:12px;">
                    {pred["class_name"]}
                </div>
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-size:0.8rem; color:#94a3b8;">Confidence</span>
                    <span style="font-size:1.1rem; font-weight:700; color:{badge_color};">
                        {conf_pct:.1f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(pred["confidence"])

            # Top-3 Plotly bar chart
            top3      = pred["top3"]
            df_top    = pd.DataFrame(top3, columns=["Disease", "Probability"])
            df_top["Probability_%"] = (df_top["Probability"] * 100).round(2)

            fig = go.Figure(go.Bar(
                x=df_top["Probability_%"],
                y=df_top["Disease"],
                orientation="h",
                marker=dict(
                    color=df_top["Probability_%"],
                    colorscale=[[0, "#164e2a"], [0.5, "#22c55e"], [1, "#86efac"]],
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                ),
                text=[f"{p:.1f}%" for p in df_top["Probability_%"]],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=13, family="Inter"),
                hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=0, r=40, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 115],
                           showgrid=True,
                           gridcolor="rgba(255,255,255,0.05)",
                           tickfont=dict(color="#64748b", size=11),
                           zeroline=False),
                yaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            # Timestamp
            st.caption(f"🕐 Analysed at {pred['timestamp']}")

            # Download annotated result
            img_buf = build_download_image(pred)
            fname   = f"rice_{pred['class_name'].lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            st.download_button(
                label="📥 Download Annotated Result",
                data=img_buf,
                file_name=fname,
                mime="image/png",
                width="stretch",
            )

        else:
            # Empty state
            lottie_wait = load_lottie(LOTTIE_EMPTY)
            if lottie_wait:
                st_lottie(lottie_wait, height=180, key="lottie_wait")
            else:
                st.markdown("""
                <div style="height:160px; display:flex; align-items:center;
                            justify-content:center; font-size:3rem; opacity:.35;">📊</div>
                """, unsafe_allow_html=True)
            st.info("Upload an image on the left and press **🔍 Predict Disease** to see results here.",
                    icon="👈")

        st.markdown('</div>', unsafe_allow_html=True)   # close glass-card


# ══════════════════════════════════════════════
# ③  HISTORY PAGE
# ══════════════════════════════════════════════
elif selected == "📜 History":
    st.markdown('<h1 class="main-header" style="font-size:2.4rem;">📜 Prediction History</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="margin-bottom:1.2rem;">Review your past analyses</p>',
                unsafe_allow_html=True)

    if not st.session_state.history:
        lottie_empty = load_lottie(LOTTIE_EMPTY)
        if lottie_empty:
            st_lottie(lottie_empty, height=260, key="lottie_empty_hist")
        st.info("No predictions yet. Head over to the **🔬 Classifier** to get started!", icon="📭")
    else:
        history_rev = list(reversed(st.session_state.history))

        # ── Summary metrics ──
        diseases = [h["disease"] for h in st.session_state.history]
        avg_conf = np.mean([h["confidence"] for h in st.session_state.history])
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Analyses", len(st.session_state.history))
        m2.metric("Avg Confidence", f"{avg_conf:.1f}%")
        m3.metric("Unique Diseases", len(set(diseases)))

        # ── Distribution donut ──
        st.markdown("<br>", unsafe_allow_html=True)
        dist_col, table_col = st.columns([1, 1.4], gap="large")

        with dist_col:
            from collections import Counter
            counts = Counter(diseases)
            fig_pie = go.Figure(go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.55,
                marker=dict(
                    colors=["#22c55e", "#86efac", "#4ade80", "#bbf7d0", "#166534"],
                    line=dict(color="#0f172a", width=2),
                ),
                textfont=dict(color="#e2e8f0", size=12),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            ))
            fig_pie.add_annotation(text=f"<b>{len(st.session_state.history)}</b><br><span>analyses</span>",
                                   x=0.5, y=0.5, showarrow=False,
                                   font=dict(size=16, color="#86efac", family="Space Grotesk"))
            fig_pie.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

        with table_col:
            df_hist = pd.DataFrame([
                {"#":       i + 1,
                 "Timestamp": h["timestamp"],
                 "Disease":   h["disease"],
                 "Confidence (%)": h["confidence"]}
                for i, h in enumerate(history_rev)
            ])
            st.dataframe(
                df_hist,
                width="stretch",
                hide_index=True,
                column_config={
                    "Confidence (%)": st.column_config.ProgressColumn(
                        "Confidence", format="%.1f%%", min_value=0, max_value=100,
                    ),
                },
            )

        # ── Card-style recent entries ──
        st.markdown("<br>**Recent Analyses**")
        for h in history_rev[:5]:
            conf = h["confidence"]
            bar_color = "#22c55e" if conf >= 70 else "#f59e0b" if conf >= 45 else "#ef4444"
            top3_text = " · ".join(
                f"{lbl} ({p*100:.0f}%)" for lbl, p in h.get("top3", [])
            )
            st.markdown(f"""
            <div class="history-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="disease-badge">{h["disease"]}</span>
                    <span style="font-size:0.78rem; color:#475569;">{h["timestamp"]}</span>
                </div>
                <div style="margin-top:8px; height:4px; background:rgba(255,255,255,0.06);
                            border-radius:10px; overflow:hidden;">
                    <div style="height:100%; width:{conf}%; background:{bar_color};
                                border-radius:10px; transition:width 1s ease;"></div>
                </div>
                <div style="font-size:0.75rem; color:#64748b; margin-top:6px;">
                    Top 3: {top3_text if top3_text else "—"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All History", width="content"):
            st.session_state.history = []
            st.session_state.current_prediction = None
            st.success("History cleared successfully.")
            st.rerun()


# ══════════════════════════════════════════════
# ④  ABOUT PAGE
# ══════════════════════════════════════════════
elif selected == "ℹ️ About":
    st.markdown('<h1 class="main-header" style="font-size:2.4rem;">ℹ️ About This App</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="margin-bottom:1.5rem;">Model details · Feature pipeline · Tech stack</p>',
                unsafe_allow_html=True)

    top_l, top_r = st.columns([1.3, 1], gap="large")

    with top_l:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#86efac; margin-top:0; font-family:'Space Grotesk',sans-serif;">
                🤖 Model Details
            </h3>
            <table style="width:100%; border-collapse:collapse; font-size:0.88rem; color:#e2e8f0;">
                <tr>
                    <td style="padding:8px 0; color:#64748b; width:40%;">Architecture</td>
                    <td>Traditional ML (scikit-learn)</td>
                </tr>
                <tr>
                    <td style="padding:8px 0; color:#64748b;">Input Size</td>
                    <td>256 × 256 px (resized)</td>
                </tr>
                <tr>
                    <td style="padding:8px 0; color:#64748b;">Feature Dims</td>
                    <td>100+ handcrafted features</td>
                </tr>
                <tr>
                    <td style="padding:8px 0; color:#64748b;">Preprocessing</td>
                    <td>Imputation → StandardScaler → Feature selection</td>
                </tr>
                <tr>
                    <td style="padding:8px 0; color:#64748b;">Output</td>
                    <td>Disease class + probability distribution</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Feature modules
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#a5b4fc; margin-top:0;">📦 Feature Extraction Pipeline</h4>
        """, unsafe_allow_html=True)

        features_info = [
            ("🎨", "Color Features",    "RGB · HSV · LAB mean/std/skew + 8-bin histograms"),
            ("🔳", "GLCM Texture",      "Contrast · Correlation · Energy · Homogeneity at 2 distances × 4 angles"),
            ("🔁", "LBP Patterns",      "Uniform LBP histogram + mean/std (radius=1, 8 points)"),
            ("📐", "Shape Features",    "Leaf area · perimeter · aspect ratio · solidity · eccentricity"),
            ("🌀", "HOG Gradients",     "9 orientations, 32×32 cells, 2×2 blocks + Sobel gradient stats"),
            ("🔴", "Lesion Detection",  "Lesion area · ratio · count · mean/largest area via HSV+LAB thresholding"),
        ]
        for icon, name, desc in features_info:
            st.markdown(f"""
            <div style="display:flex; gap:12px; padding:8px 0;
                        border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:1.3rem; flex-shrink:0;">{icon}</span>
                <div>
                    <div style="font-weight:600; color:#e2e8f0; font-size:0.88rem;">{name}</div>
                    <div style="color:#64748b; font-size:0.78rem; margin-top:2px;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with top_r:
        # Tech stack
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#86efac; margin-top:0;">🛠️ Tech Stack</h4>
        """, unsafe_allow_html=True)

        stack = [
            ("🟢", "Streamlit",          "Web framework"),
            ("📊", "Plotly",             "Interactive charts"),
            ("🔵", "OpenCV",             "Image processing"),
            ("🟡", "scikit-image",       "GLCM · LBP · HOG · RegionProps"),
            ("🟠", "scikit-learn",       "Classifier · Scaler · Imputer"),
            ("🖼️",  "Pillow",            "Image annotation & export"),
            ("🎬", "streamlit-lottie",   "Animated UI elements"),
            ("🧭", "streamlit-option-menu", "Sidebar navigation"),
        ]
        for icon, name, role in stack:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:7px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#e2e8f0; font-size:0.85rem;">{icon} {name}</span>
                <span style="color:#4b5563; font-size:0.75rem;">{role}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Disease classes
        st.markdown("""
        <div class="glass-card" style="margin-top:0;">
            <h4 style="color:#86efac; margin-top:0;">🌿 Disease Classes</h4>
            <p style="color:#94a3b8; font-size:0.82rem; line-height:1.7;">
                The model is trained to identify the following rice leaf conditions.
                The exact classes depend on your training dataset labels.
            </p>
        </div>
        """, unsafe_allow_html=True)

        try:
            classes = list(le.classes_)
            for cls in classes:
                st.markdown(f'<span class="feature-tag">🌾 {cls}</span>', unsafe_allow_html=True)
        except Exception:
            st.caption("Class labels will appear here after model is loaded.")

    # ── Bottom: how-to tips ──
    st.markdown("<br>### 💡 Tips for Best Results")
    t1, t2, t3 = st.columns(3)
    tips = [
        ("📸", "Good Lighting",  "Use natural daylight or even indoor light — avoid heavy shadows on the leaf."),
        ("🌿", "Clear Leaf",     "Ensure the entire leaf is visible with minimal background clutter."),
        ("🔍", "Close-Up Shot",  "Get close enough so the leaf fills most of the frame (80%+ coverage)."),
    ]
    for col, (icon, title, tip) in zip([t1, t2, t3], tips):
        col.markdown(f"""
        <div class="about-feature">
            <div style="font-size:2rem; margin-bottom:8px;">{icon}</div>
            <h4>{title}</h4>
            <p>{tip}</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#1e293b; font-size:0.72rem;
            border-top:1px solid rgba(255,255,255,0.05);
            padding-top:14px; margin-top:12px; letter-spacing:.05em;">
    🌾 Rice Leaf Disease Classifier · Built with Streamlit, OpenCV &amp; Plotly
</div>
""", unsafe_allow_html=True)

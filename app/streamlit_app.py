import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import warnings

# Optional: silence Streamlit future warnings during demo
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="YouTube Ad Revenue Predictor", layout="centered")
st.title("YouTube Ad Revenue Predictor 💸")
st.caption("Predict ad revenue (USD) for a video using engagement & audience signals.")

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(path):
        st.error("model.pkl not found in the app folder. Place your trained model here.")
        st.stop()
    return joblib.load(path)

model = load_model()

st.sidebar.header("Video Performance Inputs")

# Numeric inputs
views = st.sidebar.number_input("Views", min_value=0, value=1000, step=100)
likes = st.sidebar.number_input("Likes", min_value=0, value=100, step=10)
comments = st.sidebar.number_input("Comments", min_value=0, value=20, step=5)
watch_time_minutes = st.sidebar.number_input("Watch Time (minutes)", min_value=0, value=500, step=50)
video_length_minutes = st.sidebar.number_input("Video Length (minutes)", min_value=1, value=10, step=1)
subscribers = st.sidebar.number_input("Channel Subscribers", min_value=0, value=10000, step=100)

# Categorical inputs
category = st.sidebar.text_input("Category", value="Education")
device = st.sidebar.selectbox("Primary Device", ["Mobile", "Desktop", "TV", "Tablet"])
country = st.sidebar.text_input("Country (ISO-like code)", value="IN")

# Derived feature (guard against divide-by-zero)
engagement_rate = (likes + comments) / views if views > 0 else 0.0

# Exact order used in training
row = {
    "views": views,
    "likes": likes,
    "comments": comments,
    "watch_time_minutes": watch_time_minutes,
    "video_length_minutes": video_length_minutes,
    "subscribers": subscribers,
    "engagement_rate": engagement_rate,
    "category": category,
    "device": device,
    "country": country,
}
X = pd.DataFrame([row])

st.subheader("Your Input")
st.dataframe(X, width="stretch")  # updated (no use_container_width)

if st.button("Predict Revenue (USD)"):
    if X.loc[0, "views"] <= 0:
        st.warning("Views must be > 0 to compute engagement_rate. Please enter a positive number.")
    else:
        try:
            pred = float(model.predict(X)[0])
            rpm = (pred / X.loc[0, "views"]) * 1000
            st.success(f"Estimated Ad Revenue: ${pred:,.2f} USD")
            st.caption(f"Approx. RPM: ${rpm:,.2f} per 1,000 views (ballpark)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with st.expander("About this app"):
    st.write(
        "- Trained with a sklearn Pipeline: imputers + StandardScaler (numeric) + OneHotEncoder (categorical) + final regressor (e.g., Lasso).\\n"
        "- Required columns: views, likes, comments, watch_time_minutes, video_length_minutes, subscribers, engagement_rate, category, device, country.\\n"
        "- Place the serialized pipeline as **model.pkl** inside this **app/** folder."
    )
# --- Top drivers panel (safe path handling) ---
from pathlib import Path
import pandas as pd

try:
    base_dir = Path(__file__).resolve().parent          # app/
except NameError:
    base_dir = Path.cwd()                               # notebook fallback

feat_path = (base_dir / ".." / "model_top_features.csv").resolve()

st.markdown("---")
st.subheader("What drives revenue (top features)")

if feat_path.exists():
    tf = pd.read_csv(feat_path)
    st.dataframe(tf.head(20), width="stretch")
    st.download_button(
        "Download top-features CSV",
        tf.to_csv(index=False).encode("utf-8"),
        "model_top_features.csv",
        "text/csv",
    )
else:
    st.info(f"Top-features file not found at:\n{feat_path}")
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Page setup ----------
st.set_page_config(page_title="YouTube Ad Revenue Predictor", layout="wide", page_icon="💸")
# ---- THEME PICKER with Pastel default ----
import streamlit as st

THEMES = {
    "Pastel Breeze": {
        "bg_top":    "#FDF7FF",  # very light lilac
        "bg_bottom": "#F7FAFF",  # very light blue
        "rad1": "rgba(255,182,193,0.24)",  # light pink
        "rad2": "rgba(173,216,230,0.22)",  # light blue
        "sidebar_bg": "linear-gradient(180deg, rgba(255,255,255,0.95), rgba(247,250,255,0.90))",
        "text": "#374151",  # slate-700
        "card_bg": "rgba(255,255,255,0.92)",
        "card_border": "rgba(55,65,81,0.10)",
        "kpi_bg": "rgba(255,255,255,0.98)",
        "kpi_border": "rgba(55,65,81,0.12)",
        "accent": "#F472B6",  # pink-400
        "accent_hover": "#F9A8D4"  # pink-300
    },
    # keep your other themes here if you like…
}

st.sidebar.header("⚙️ Controls")
theme_names = list(THEMES.keys())
default_index = theme_names.index("Pastel Breeze")
theme_name = st.sidebar.selectbox("🎨 Theme", theme_names, index=default_index)
T = THEMES[theme_name]

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(900px 520px at 8% 0%, {T['rad1']} 0%, rgba(0,0,0,0) 42%),
    radial-gradient(700px 420px at 92% 8%, {T['rad2']} 0%, rgba(0,0,0,0) 42%),
    linear-gradient(180deg, {T['bg_top']} 0%, {T['bg_bottom']} 100%);
}}
[data-testid="stSidebar"] {{ background: {T['sidebar_bg']}; }}
.card {{
  background: {T['card_bg']};
  border: 1px solid {T['card_border']};
  border-radius: 16px; padding: 18px; box-shadow: 0 8px 22px rgba(0,0,0,0.08);
  backdrop-filter: blur(6px);
}}
.kpi {{
  display:flex; align-items:center; gap:12px; padding:12px 14px; border-radius:14px;
  background:{T['kpi_bg']}; border:1px solid {T['kpi_border']}; color:{T['text']};
}}
/* Typography & links */
h1,h2,h3,h4,h5,h6,p,span,div,label,small,code {{ color:{T['text']} !important; }}
a {{ color:{T['accent']} !important; }}
/* Buttons → pastel */
div.stButton > button {{
  background:{T['accent']}; color:#1F2937; border:0; border-radius:12px; padding:0.6rem 1rem;
}}
div.stButton > button:hover {{ background:{T['accent_hover']}; }}
/* Inputs focus ring */
input:focus, textarea:focus, select:focus {{ outline: 2px solid {T['accent']}33; }}
#MainMenu, footer, header {{ visibility:hidden; }}
.block-container {{ padding-top:1.2rem; }}
</style>
""", unsafe_allow_html=True)
# ---- END THEME PICKER ----

# ---------- Global styles (gradient bg + glass cards + hide footer/menu) ----------
st.markdown("""
<style>
/* App background */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1000px 600px at 10% 0%, rgba(108,92,231,0.18) 0%, rgba(15,17,26,0.0) 40%),
              radial-gradient(800px 500px at 90% 10%, rgba(16,185,129,0.18) 0%, rgba(15,17,26,0.0) 40%),
              linear-gradient(180deg, #0b1220 0%, #0b1326 100%);
}
/* Sidebar background */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(17,24,39,0.85));
}
/* Glass card look */
.block-container { padding-top: 1.2rem; }
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px; padding: 18px; box-shadow: 0 6px 20px rgba(0,0,0,0.25);
  backdrop-filter: blur(8px);
}
.kpi {
  display: flex; align-items: center; gap: 12px; padding: 12px 14px; border-radius: 14px;
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
}
.small { opacity: 0.8; font-size: 0.9rem; }
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Model loader ----------
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(path):
        st.error("model.pkl not found in the app folder. Place your trained model here.")
        st.stop()
    return joblib.load(path)

model = load_model()

# Try to extract category lists from the trained pipeline for nice dropdowns
def get_category_options(_model):
    try:
        ohe = _model.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
        cats = ohe.categories_
        return {
            "category": list(cats[0]),
            "device": list(cats[1]),
            "country": list(cats[2]),
        }
    except Exception:
        # Fallbacks if extraction fails
        return {
            "category": ["Education","Entertainment","Music","Science","Gaming","News"],
            "device": ["Mobile","Desktop","TV","Tablet"],
            "country": ["IN","US","GB","AE","SG","AU"],
        }

opts = get_category_options(model)

# ---------- Title row ----------
c1, c2 = st.columns([0.75, 0.25], gap="large")
with c1:
    st.markdown("<div class='card'><h2>💸 YouTube Ad Revenue Predictor</h2><p class='small'>Instant USD estimate from early performance signals — views, engagement & audience mix.</p></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><b>Status:</b> <span style='color:#22c55e'>Model Loaded</span><br><span class='small'>Pipeline: impute → scale → OHE → Lasso</span></div>", unsafe_allow_html=True)

st.markdown("")  # spacing

# ---------- Sidebar (Presets + About) ----------
st.sidebar.header("⚙️ Controls")
preset = st.sidebar.selectbox(
    "Quick preset",
    ["Custom","New Upload (small channel)","Growing Video","Viral Spike"],
    index=0
)

# Preset defaults
preset_values = {
    "Custom": {},
    "New Upload (small channel)": dict(views=1500, likes=80, comments=15, watch_time_minutes=1200, video_length_minutes=8, subscribers=5000, category="Education", device="Mobile", country="IN"),
    "Growing Video": dict(views=12000, likes=900, comments=180, watch_time_minutes=25000, video_length_minutes=12, subscribers=150000, category="Entertainment", device="Mobile", country="IN"),
    "Viral Spike": dict(views=250000, likes=18000, comments=3500, watch_time_minutes=550000, video_length_minutes=10, subscribers=1200000, category="Music", device="Mobile", country="US"),
}

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use **Batch Predict** tab to upload a CSV of multiple videos.")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📦 Batch Predict", "📈 Insights"])

with tab1:
    # Layout: inputs (left) and results (right)
    left, right = st.columns([0.55, 0.45], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Inputs")
        with st.form("single_predict_form", clear_on_submit=False):
            pv = preset_values.get(preset, {})
            views = st.number_input("Views", min_value=0, value=pv.get("views", 1000), step=100)
            likes = st.number_input("Likes", min_value=0, value=pv.get("likes", 100), step=10)
            comments = st.number_input("Comments", min_value=0, value=pv.get("comments", 20), step=5)
            watch_time_minutes = st.number_input("Watch Time (minutes)", min_value=0, value=pv.get("watch_time_minutes", 500), step=50)
            video_length_minutes = st.number_input("Video Length (minutes)", min_value=1, value=pv.get("video_length_minutes", 10), step=1)
            subscribers = st.number_input("Channel Subscribers", min_value=0, value=pv.get("subscribers", 10000), step=100)
            category = st.selectbox("Category", options=opts["category"], index=opts["category"].index(pv.get("category", opts["category"][0])) if pv.get("category") in opts["category"] else 0)
            device = st.selectbox("Primary Device", options=opts["device"], index=opts["device"].index(pv.get("device", opts["device"][0])) if pv.get("device") in opts["device"] else 0)
            country = st.selectbox("Country", options=opts["country"], index=opts["country"].index(pv.get("country", opts["country"][0])) if pv.get("country") in opts["country"] else 0)

            submitted = st.form_submit_button("Predict Revenue (USD)")

        # Prepare row
        engagement_rate = (likes + comments) / views if views > 0 else 0.0
        X = pd.DataFrame([{
            "views": views,
            "likes": likes,
            "comments": comments,
            "watch_time_minutes": watch_time_minutes,
            "video_length_minutes": video_length_minutes,
            "subscribers": subscribers,
            "engagement_rate": engagement_rate,
            "category": category,
            "device": device,
            "country": country,
        }])

        st.caption("Input preview")
        st.dataframe(X, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Results")
        if submitted:
            if views <= 0:
                st.warning("Views must be > 0 to compute engagement_rate.")
            else:
                try:
                    pred = float(model.predict(X)[0])
                    rpm = (pred / max(views, 1)) * 1000.0  # per 1000 views
                    c1, c2 = st.columns(2, gap="large")
                    with c1:
                        st.markdown(f"<div class='kpi'><span>💰</span><div><b>Estimated Revenue</b><br>${pred:,.2f} USD</div></div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div class='kpi'><span>📈</span><div><b>RPM (per 1,000 views)</b><br>${rpm:,.2f}</div></div>", unsafe_allow_html=True)

                    # Friendly hints
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Suggestions:**")
                    st.write("- Boost **watch time** (chapters, hooks, pattern interrupts).")
                    st.write("- Encourage **comments** (ask a 1-line question in the first 30 seconds).")
                    st.write("- Double-check **country/device** targeting for sponsors and ad types.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("Fill inputs on the left and click **Predict**.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Batch Predict (CSV)")
    st.caption("Upload rows matching the model columns. If `engagement_rate` is missing, we'll compute it from likes + comments + views (safe for zero views).")

    # Downloadable template
    template = pd.DataFrame([{
        "views": 1000, "likes": 100, "comments": 20,
        "watch_time_minutes": 500, "video_length_minutes": 10,
        "subscribers": 10000, "engagement_rate": 0.12,
        "category": "Education", "device": "Mobile", "country": "IN"
    }])
    st.download_button(
        "Download CSV Template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="batch_template.csv",
        mime="text/csv"
    )

    csv = st.file_uploader("Upload CSV", type=["csv"])
    if csv is not None:
        try:
            df_in = pd.read_csv(csv)
            # compute engagement_rate if missing
            if "engagement_rate" not in df_in.columns and {"likes","comments","views"}.issubset(df_in.columns):
                safe_views = df_in["views"].replace(0, np.nan)
                df_in["engagement_rate"] = (df_in["likes"] + df_in["comments"]) / safe_views
                df_in["engagement_rate"] = df_in["engagement_rate"].fillna(0.0)

            preds = model.predict(df_in)
            out = df_in.copy()
            out["pred_ad_revenue_usd"] = preds
            st.dataframe(out.head(50), width="stretch")
            st.download_button("Download Predictions", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Top Drivers (from training)")
    # Load the feature importance/coefficients CSV you saved in Day 2
    try:
        base_dir = Path(__file__).resolve().parent  # app/
    except NameError:
        base_dir = Path.cwd()                       # notebook fallback
    feat_path = (base_dir / ".." / "model_top_features.csv").resolve()

    if feat_path.exists():
        tf = pd.read_csv(feat_path)
        st.dataframe(tf.head(25), width="stretch")
    else:
        st.info(f"Feature file not found at: {feat_path}")
    st.markdown("</div>", unsafe_allow_html=True)
    # --- FINAL LIGHT PASTEL OVERRIDE ---
st.markdown("""
<style>
/* App body (very light lilac) */
[data-testid="stAppViewContainer"] {
  background: #FDF7FF !important;
}
/* Sidebar (white) */
[data-testid="stSidebar"] {
  background: #FFFFFF !important;
}
/* Card & KPI blocks (soft white) */
.card {
  background: rgba(255,255,255,0.95) !important;
  border: 1px solid rgba(55,65,81,0.10) !important;
}
.kpi {
  background: rgba(255,255,255,0.98) !important;
  border: 1px solid rgba(55,65,81,0.12) !important;
}
/* Text color (slate) so it doesn't look gray on white */
h1,h2,h3,h4,h5,h6,p,span,div,label,small,code {
  color: #374151 !important;
}
/* Buttons in a pastel pink */
div.stButton > button {
  background:#F472B6 !important; color:#1F2937 !important; border:0 !important;
  border-radius:12px !important; padding:0.6rem 1rem !important;
}
div.stButton > button:hover { background:#F9A8D4 !important; }
/* Hide default chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


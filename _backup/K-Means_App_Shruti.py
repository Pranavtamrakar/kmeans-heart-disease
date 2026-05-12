import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioCluster | Heart Disease Pattern Discovery",
    page_icon="❤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. CUSTOM CSS — Dark Clinical Cardiology Theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@600;700;900&display=swap');

    /* ---------- Base App ---------- */
    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(220, 38, 38, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 80% 100%, rgba(244, 63, 94, 0.06) 0%, transparent 40%),
            linear-gradient(180deg, #0a0e1a 0%, #0f1420 100%);
        color: #e5e7eb;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1220 0%, #14192b 100%);
        border-right: 1px solid rgba(220, 38, 38, 0.15);
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f43f5e !important;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #1a2036 !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(244, 63, 94, 0.25) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ---------- Hero Header ---------- */
    .hero-wrap {
        text-align: center;
        padding: 1.5rem 0 0.8rem 0;
        position: relative;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 54px;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #f43f5e 50%, #dc2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 15px;
        color: #94a3b8;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.4rem;
    }
    .hero-tagline {
        font-size: 17px;
        color: #cbd5e1;
        font-style: italic;
        font-weight: 300;
        margin-top: 0.8rem;
        max-width: 720px;
        margin-left: auto;
        margin-right: auto;
    }

    /* ---------- Animated Heartbeat ---------- */
    .pulse-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 16px;
        margin: 1.2rem 0 1.5rem 0;
    }
    .heart-pulse {
        width: 48px;
        height: 48px;
        animation: beat 1.2s ease-in-out infinite;
        filter: drop-shadow(0 0 12px rgba(244, 63, 94, 0.7));
    }
    @keyframes beat {
        0%, 100% { transform: scale(1); }
        15%      { transform: scale(1.18); }
        30%      { transform: scale(1); }
        45%      { transform: scale(1.12); }
        60%      { transform: scale(1); }
    }
    .ekg-line {
        flex: 1;
        max-width: 380px;
        height: 40px;
    }
    .ekg-path {
        stroke: #f43f5e;
        stroke-width: 2;
        fill: none;
        stroke-dasharray: 600;
        stroke-dashoffset: 600;
        animation: draw 2.4s linear infinite;
        filter: drop-shadow(0 0 4px rgba(244, 63, 94, 0.6));
    }
    @keyframes draw {
        0%   { stroke-dashoffset: 600; }
        100% { stroke-dashoffset: -600; }
    }

    /* ---------- Cards ---------- */
    .card {
        background: linear-gradient(145deg, rgba(20, 25, 43, 0.85) 0%, rgba(15, 20, 32, 0.85) 100%);
        border: 1px solid rgba(244, 63, 94, 0.18);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(8px);
    }
    .card-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #f43f5e;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-title::before {
        content: "";
        width: 6px;
        height: 6px;
        background: #f43f5e;
        border-radius: 50%;
        box-shadow: 0 0 8px #f43f5e;
    }

    /* ---------- Section Subheaders ---------- */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        margin: 0.5rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .section-title-accent {
        width: 4px;
        height: 26px;
        background: linear-gradient(180deg, #dc2626, #f43f5e);
        border-radius: 2px;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #f43f5e 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 18px rgba(220, 38, 38, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 26px rgba(244, 63, 94, 0.55);
        background: linear-gradient(135deg, #f43f5e 0%, #dc2626 100%);
    }

    /* ---------- DataFrame ---------- */
    [data-testid="stDataFrame"] {
        background: rgba(20, 25, 43, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(244, 63, 94, 0.15);
    }

    /* ---------- Scrollable Summary ---------- */
    .summary-scroll {
        height: 440px;
        overflow-y: auto;
        padding-right: 8px;
    }
    .summary-scroll::-webkit-scrollbar { width: 6px; }
    .summary-scroll::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #dc2626, #f43f5e);
        border-radius: 3px;
    }
    .cluster-block {
        background: rgba(15, 20, 32, 0.7);
        border-left: 3px solid #f43f5e;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        border-radius: 8px;
    }
    .cluster-block h3 {
        color: #f43f5e;
        font-size: 16px;
        margin: 0 0 0.4rem 0;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .cluster-block p, .cluster-block li {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.55;
    }

    /* ---------- Result Card ---------- */
    .result-card {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.18) 0%, rgba(244, 63, 94, 0.10) 100%);
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.4rem;
    }
    .result-value {
        font-family: 'Playfair Display', serif;
        font-size: 42px;
        font-weight: 900;
        color: #ffffff;
        margin: 0;
    }
    .result-accent {
        color: #f43f5e;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        margin-top: 2.5rem;
        padding: 1rem;
        border-top: 1px solid rgba(244, 63, 94, 0.12);
        letter-spacing: 1px;
    }
    .footer span { color: #f43f5e; }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 3. LOAD MODELS, DATA & IMAGES
# ─────────────────────────────────────────────
with open('kmeans_model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

pca_2d_df        = pd.read_excel('pca_2d.xlsx')
cluster_analysis = pd.read_excel('cluster_analysis.xlsx')

try:
    with open('cluster_summaries.pkl', 'rb') as summary_file:
        cluster_summaries = pickle.load(summary_file)
except FileNotFoundError:
    cluster_summaries = None

# Source-of-truth column schema for the trained model
with open('feature_columns.pkl', 'rb') as fc_file:
    FEATURE_COLUMNS = pickle.load(fc_file)

# ─────────────────────────────────────────────
# 4. SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <svg width="56" height="56" viewBox="0 0 24 24" fill="#f43f5e" xmlns="http://www.w3.org/2000/svg"
             style="filter: drop-shadow(0 0 12px rgba(244,63,94,0.6));">
            <path d="M12 21s-7-4.35-9.5-9.5C0.96 8.05 3.05 4 7 4c2 0 3.5 1 5 3 1.5-2 3-3 5-3 3.95 0 6.04 4.05 4.5 7.5C19 16.65 12 21 12 21z"/>
        </svg>
        <h2 style="margin: 0.4rem 0 0 0; font-family: 'Playfair Display', serif;">CardioCluster</h2>
        <p style="font-size:11px; letter-spacing:2px; text-transform:uppercase; color:#94a3b8; margin-top:0.2rem;">
            Patient Profile Input
        </p>
    </div>
    <hr style="border-color: rgba(244,63,94,0.15); margin: 0 0 1rem 0;"/>
    """,
    unsafe_allow_html=True,
)

# Patient-facing fields (id is hidden; num is rebranded as "Heart Disease Stage")
numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

categorical_options = {
    'sex':     ['Female', 'Male'],
    'dataset': ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'],
    'cp':      ['asymptomatic', 'atypical angina', 'non-anginal', 'typical angina'],
    'fbs':     ['False', 'True'],
    'restecg': ['lv hypertrophy', 'normal', 'st-t abnormality'],
    'exang':   ['False', 'True'],
    'slope':   ['downsloping', 'flat', 'upsloping'],
    'thal':    ['fixed defect', 'normal', 'reversable defect'],
    'Heart Disease Stage': ['0', '1', '2', '3', '4'],
}

# Friendlier labels for the sidebar inputs
pretty_labels = {
    'age': 'Age (years)',
    'trestbps': 'Resting BP (mm Hg)',
    'chol': 'Cholesterol (mg/dl)',
    'thalch': 'Max Heart Rate',
    'oldpeak': 'ST Depression',
    'ca': 'Major Vessels (0–3)',
    'sex': 'Sex',
    'dataset': 'Source Dataset',
    'cp': 'Chest Pain Type',
    'fbs': 'Fasting Blood Sugar > 120',
    'restecg': 'Resting ECG',
    'exang': 'Exercise-Induced Angina',
    'slope': 'ST Slope',
    'thal': 'Thalassemia',
    'Heart Disease Stage': 'Heart Disease Stage (0–4)',
}

st.sidebar.markdown("### Clinical Metrics")
user_inputs = {}
for feature in numeric_features:
    user_inputs[feature] = st.sidebar.number_input(
        pretty_labels.get(feature, feature), value=0.0, key=f"input_{feature}"
    )

st.sidebar.markdown("### Categorical Factors")
for feature, options in categorical_options.items():
    selected_value = st.sidebar.selectbox(
        pretty_labels.get(feature, feature), options, key=f"select_{feature}"
    )
    for option in options:
        user_inputs[f"{feature}_{option}"] = 1 if selected_value == option else 0

# Build a model-ready DataFrame that matches feature_columns.pkl exactly.
# Hidden columns (id, restecg_0, slope_0, thal_0) default to 0.
# "Heart Disease Stage" choice -> the model's 'num' column.
model_row = {col: 0 for col in FEATURE_COLUMNS}

for feature in numeric_features:
    if feature in model_row:
        model_row[feature] = user_inputs[feature]

hds_choice = next(
    (opt for opt in categorical_options['Heart Disease Stage']
     if user_inputs.get(f'Heart Disease Stage_{opt}') == 1),
    '0',
)
if 'num' in model_row:
    model_row['num'] = float(hds_choice)

for feature in categorical_options:
    if feature == 'Heart Disease Stage':
        continue
    for option in categorical_options[feature]:
        key = f"{feature}_{option}"
        if key in model_row:
            model_row[key] = user_inputs[key]

input_df = pd.DataFrame([model_row], columns=FEATURE_COLUMNS)

# ─────────────────────────────────────────────
# 5. MAIN PAGE — HERO
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-wrap">
        <p class="hero-sub">Unsupervised Cardiovascular Risk Patterns</p>
        <h1 class="hero-title">CardioCluster</h1>
        <p class="hero-tagline">
            Discovering hidden patient subgroups across 920 hearts using K-Means clustering,
            PCA, and locally-hosted LLaMA 3 narrative analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Animated heartbeat + EKG line
st.markdown(
    """
    <div class="pulse-wrap">
        <svg class="ekg-line" viewBox="0 0 400 40" preserveAspectRatio="none">
            <path class="ekg-path"
                  d="M0,20 L80,20 L95,20 L105,8 L115,32 L125,12 L135,28 L150,20 L400,20"/>
        </svg>
        <svg class="heart-pulse" viewBox="0 0 24 24" fill="#f43f5e" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 21s-7-4.35-9.5-9.5C0.96 8.05 3.05 4 7 4c2 0 3.5 1 5 3 1.5-2 3-3 5-3 3.95 0 6.04 4.05 4.5 7.5C19 16.65 12 21 12 21z"/>
        </svg>
        <svg class="ekg-line" viewBox="0 0 400 40" preserveAspectRatio="none">
            <path class="ekg-path"
                  d="M0,20 L250,20 L265,8 L275,32 L285,12 L295,28 L310,20 L400,20"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 6. TOP STATS STRIP
# ─────────────────────────────────────────────
stat1, stat2, stat3, stat4 = st.columns(4)
stat_template = """
<div class="card" style="text-align:center; padding: 1rem 0.5rem;">
    <p style="font-size:11px; color:#94a3b8; letter-spacing:2px; text-transform:uppercase; margin:0;">{label}</p>
    <p style="font-family:'Playfair Display', serif; font-size:32px; font-weight:800; color:#ffffff; margin:0.3rem 0 0 0;">
        <span style="color:#f43f5e;">{value}</span>
    </p>
</div>
"""
with stat1: st.markdown(stat_template.format(label="Patients Analyzed", value="920"), unsafe_allow_html=True)
with stat2: st.markdown(stat_template.format(label="Risk Clusters",     value="5"),   unsafe_allow_html=True)
with stat3: st.markdown(stat_template.format(label="Clinical Features", value="15"),  unsafe_allow_html=True)
with stat4: st.markdown(stat_template.format(label="Source Hospitals",  value="4"),   unsafe_allow_html=True)

st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. ROW 1 — PCA scatter | Cluster Summaries
# ─────────────────────────────────────────────
left_col, right_col = st.columns([1.15, 1])

with left_col:
    st.markdown(
        '<div class="section-title"><span class="section-title-accent"></span>PCA Cluster Map</div>',
        unsafe_allow_html=True,
    )
    # Custom cardiology-tuned color sequence
    heart_palette = ['#f43f5e', '#fb7185', '#fda4af', '#dc2626', '#9f1239']

    fig = px.scatter(
        pca_2d_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
        color_discrete_sequence=heart_palette,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color='rgba(255,255,255,0.4)')))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,20,32,0.4)',
        font=dict(family='Inter, sans-serif', color='#e5e7eb', size=12),
        title=dict(text='', x=0.5),
        xaxis=dict(gridcolor='rgba(244,63,94,0.08)', zerolinecolor='rgba(244,63,94,0.15)'),
        yaxis=dict(gridcolor='rgba(244,63,94,0.08)', zerolinecolor='rgba(244,63,94,0.15)'),
        legend=dict(
            bgcolor='rgba(20,25,43,0.6)',
            bordercolor='rgba(244,63,94,0.2)',
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown(
        '<div class="section-title"><span class="section-title-accent"></span>Cluster Narratives</div>',
        unsafe_allow_html=True,
    )

    if isinstance(cluster_summaries, str):
        with st.container(height=440):
            st.markdown(f"<div class='cluster-block'>{cluster_summaries}</div>", unsafe_allow_html=True)

    elif isinstance(cluster_summaries, dict):
        with st.container(height=440):
            for key, value in cluster_summaries.items():
                st.markdown(
                    f"<div class='cluster-block'><h3>♥ Cluster {key}</h3>{value}</div>",
                    unsafe_allow_html=True,
                )

    else:
        st.markdown(
            "<div class='cluster-block'><h3>No summaries loaded</h3>"
            "<p>Generate them locally with <code>python generate_summary.py</code> after starting Ollama.</p></div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# 8. ROW 2 — Cluster Me | Cluster Analysis Table
# ─────────────────────────────────────────────
st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
second_left_col, second_right_col = st.columns([1, 1.2])

with second_left_col:
    st.markdown(
        '<div class="section-title"><span class="section-title-accent"></span>Profile Diagnosis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="card">
            <div class="card-title">How it works</div>
            <p style="color:#cbd5e1; font-size:14px; line-height:1.6; margin:0;">
                Enter clinical metrics in the sidebar, then click below. The trained K-Means model
                will assign your profile to one of five cardiovascular risk clusters.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    diagnose = st.button("❤  Diagnose My Cluster")
    if diagnose:
        try:
            cluster_id = int(kmeans.predict(scaler.transform(input_df))[0])
            st.markdown(
                f"""
                <div class="result-card">
                    <p class="result-label">Predicted Cardiovascular Group</p>
                    <p class="result-value">Cluster <span class="result-accent">{cluster_id}</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")

with second_right_col:
    st.markdown(
        '<div class="section-title"><span class="section-title-accent"></span>Cluster Analysis Table</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(cluster_analysis.head(), height=300, use_container_width=True)

# ─────────────────────────────────────────────
# 9. FOOTER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        Built with <span>♥</span> using Python · Scikit-learn · Streamlit · LLaMA 3 (Ollama)
        &nbsp;·&nbsp; UCI Heart Disease Dataset (920 patients)
    </div>
    """,
    unsafe_allow_html=True,
)

# Run with: streamlit run K-Means_App_Shruti.py

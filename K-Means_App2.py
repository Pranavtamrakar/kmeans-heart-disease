import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
# 1. PAGE CONFIG — must be the FIRST st call
# ─────────────────────────────────────────────
st.set_page_config(layout="wide")

# ─────────────────────────────────────────────
# 2. CUSTOM CSS — Dark Clinical Cardiology Theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@600;700;900&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(220, 38, 38, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 80% 100%, rgba(244, 63, 94, 0.06) 0%, transparent 40%),
            linear-gradient(180deg, #0a0e1a 0%, #0f1420 100%);
        color: #e5e7eb;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1220 0%, #14192b 100%);
        border-right: 1px solid rgba(220, 38, 38, 0.15);
    }
    [data-testid="stSidebar"] * { color: #e5e7eb !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f43f5e !important; font-weight: 700;
    }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #1a2036 !important; color: #e5e7eb !important;
        border: 1px solid rgba(244, 63, 94, 0.25) !important; border-radius: 8px !important;
    }
    [data-testid="stSidebar"] label {
        font-size: 13px !important; font-weight: 500 !important;
        color: #cbd5e1 !important; letter-spacing: 0.3px;
    }

    /* Title */
    .title {
        font-family: 'Playfair Display', serif;
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #f43f5e 50%, #dc2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin: 0.5rem 0 0.8rem 0;
    }
    .subheader {
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin: 0.5rem 0 0.8rem 0;
        display: flex; align-items: center; gap: 12px;
    }
    .subheader::before {
        content: ""; width: 4px; height: 24px;
        background: linear-gradient(180deg, #dc2626, #f43f5e);
        border-radius: 2px;
    }

    /* Heartbeat animation */
    .pulse-wrap {
        display: flex; justify-content: center; align-items: center;
        gap: 16px; margin: 0.5rem 0 1.5rem 0;
    }
    .heart-pulse {
        width: 48px; height: 48px;
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
    .ekg-line { flex: 1; max-width: 380px; height: 40px; }
    .ekg-path {
        stroke: #f43f5e; stroke-width: 2; fill: none;
        stroke-dasharray: 600; stroke-dashoffset: 600;
        animation: draw 2.4s linear infinite;
        filter: drop-shadow(0 0 4px rgba(244, 63, 94, 0.6));
    }
    @keyframes draw {
        0%   { stroke-dashoffset: 600; }
        100% { stroke-dashoffset: -600; }
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(145deg, rgba(20, 25, 43, 0.85) 0%, rgba(15, 20, 32, 0.85) 100%);
        border: 1px solid rgba(244, 63, 94, 0.18);
        border-radius: 16px;
        padding: 1rem 0.5rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    .stat-label {
        font-size: 11px; color: #94a3b8;
        letter-spacing: 2px; text-transform: uppercase; margin: 0;
    }
    .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 32px; font-weight: 800; color: #f43f5e;
        margin: 0.3rem 0 0 0;
    }

    /* Cluster summary cards */
    .cluster-block {
        background: rgba(15, 20, 32, 0.7);
        border-left: 3px solid #f43f5e;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        border-radius: 8px;
    }
    .cluster-block h3 {
        color: #f43f5e; font-size: 16px;
        margin: 0 0 0.4rem 0; font-weight: 700;
    }
    .cluster-block p, .cluster-block li {
        color: #cbd5e1; font-size: 14px; line-height: 1.55;
    }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #dc2626, #f43f5e); border-radius: 3px;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #f43f5e 100%);
        color: white !important; border: none;
        padding: 0.75rem 2rem; border-radius: 12px;
        font-weight: 700; font-size: 15px; letter-spacing: 1px;
        text-transform: uppercase; width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 18px rgba(220, 38, 38, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 26px rgba(244, 63, 94, 0.55);
    }

    /* Result card (replaces st.success styling) */
    div[data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.18) 0%, rgba(244, 63, 94, 0.10) 100%) !important;
        border: 1px solid rgba(244, 63, 94, 0.4) !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        animation: fadeIn 0.5s ease-out;
    }
    div[data-testid="stAlert"] * { color: #ffffff !important; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        background: rgba(20, 25, 43, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(244, 63, 94, 0.15);
    }

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

pca_2d_df      = pd.read_excel('pca_2d.xlsx')
cluster_analysis = pd.read_excel('cluster_analysis.xlsx')

try:
    with open('cluster_summaries.pkl', 'rb') as summary_file:
        cluster_summaries = pickle.load(summary_file)
except FileNotFoundError:
    cluster_summaries = None
    st.error("Cluster summaries file not found.")

sidebar_image = Image.open('Pic1.PNG')
main_image    = Image.open('Pic2.PNG')

# ─────────────────────────────────────────────
# 4. SIDEBAR — defined ONCE
# ─────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <svg width="56" height="56" viewBox="0 0 24 24" fill="#f43f5e" xmlns="http://www.w3.org/2000/svg"
             style="filter: drop-shadow(0 0 12px rgba(244,63,94,0.6));">
            <path d="M12 21s-7-4.35-9.5-9.5C0.96 8.05 3.05 4 7 4c2 0 3.5 1 5 3 1.5-2 3-3 5-3 3.95 0 6.04 4.05 4.5 7.5C19 16.65 12 21 12 21z"/>
        </svg>
        <h2 style="margin: 0.4rem 0 0 0; font-family: 'Playfair Display', serif;">Cluster Visualization</h2>
    </div>
    <hr style="border-color: rgba(244,63,94,0.15); margin: 0 0 1rem 0;"/>
    """,
    unsafe_allow_html=True,
)

numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_options = {
    'sex':               ['Female', 'Male'],
    'dataset':           ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'],
    'cp':                ['asymptomatic', 'atypical angina', 'non-anginal', 'typical angina'],
    'fbs':               ['False', 'True'],
    'restecg':           ['lv hypertrophy', 'normal', 'st-t abnormality'],
    'exang':             ['False', 'True'],
    'slope':             ['downsloping', 'flat', 'upsloping'],
    'thal':              ['fixed defect', 'normal', 'reversable defect'],
    'Heart Disease Stage': ['0', '1', '2', '3', '4']
}

user_inputs = {}  # initialise BEFORE populating

for feature in numeric_features:
    user_inputs[feature] = st.sidebar.number_input(
        feature, value=0.0, key=f"input_{feature}"
    )

for feature, options in categorical_options.items():
    selected_value = st.sidebar.selectbox(
        feature, options, key=f"select_{feature}"
    )
    for option in options:
        user_inputs[f"{feature}_{option}"] = 1 if selected_value == option else 0

input_df = pd.DataFrame([user_inputs])

# ─────────────────────────────────────────────
# 5. MAIN PAGE
# ─────────────────────────────────────────────
st.markdown('<h1 class="title">Cluster Analysis with PCA Visualization</h1>', unsafe_allow_html=True)

# Animated heartbeat + EKG lines
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

# Stats strip
stat_template = '<div class="stat-card"><p class="stat-label">{label}</p><p class="stat-value">{value}</p></div>'
s1, s2, s3, s4 = st.columns(4)
with s1: st.markdown(stat_template.format(label="Patients Analyzed", value="920"), unsafe_allow_html=True)
with s2: st.markdown(stat_template.format(label="Risk Clusters",     value="5"),   unsafe_allow_html=True)
with s3: st.markdown(stat_template.format(label="Clinical Features", value="15"),  unsafe_allow_html=True)
with s4: st.markdown(stat_template.format(label="Source Hospitals",  value="4"),   unsafe_allow_html=True)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# — Row 1: PCA scatter | Cluster Summaries —
left_col, right_col = st.columns(2)

with left_col:
    st.markdown('<h2 class="subheader">Cluster Visualization</h2>', unsafe_allow_html=True)
    heart_palette = ['#f43f5e', '#fb7185', '#fda4af', '#dc2626', '#9f1239']
    fig = px.scatter(
        pca_2d_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        title="Clusters Visualized with PCA",
        labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
        color_discrete_sequence=heart_palette,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color='rgba(255,255,255,0.4)')))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,20,32,0.4)',
        font=dict(family='Inter, sans-serif', color='#e5e7eb', size=12),
        xaxis=dict(gridcolor='rgba(244,63,94,0.08)', zerolinecolor='rgba(244,63,94,0.15)'),
        yaxis=dict(gridcolor='rgba(244,63,94,0.08)', zerolinecolor='rgba(244,63,94,0.15)'),
        legend=dict(bgcolor='rgba(20,25,43,0.6)', bordercolor='rgba(244,63,94,0.2)', borderwidth=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown('<h2 class="subheader">Cluster Summaries</h2>', unsafe_allow_html=True)

    if isinstance(cluster_summaries, str):
        # Single string — render as markdown directly
        with st.container(height=440):
            st.markdown(f"<div class='cluster-block'>{cluster_summaries}</div>", unsafe_allow_html=True)

    elif isinstance(cluster_summaries, dict):
        # Dictionary — render each cluster separately
        with st.container(height=440):
            for key, value in cluster_summaries.items():
                st.markdown(
                    f"<div class='cluster-block'><h3>♥ Cluster {key}</h3>{value}</div>",
                    unsafe_allow_html=True,
                )

    else:
        st.warning("No cluster summaries available.")

# — Row 2: Cluster Me button | Cluster Analysis Table —
second_left_col, second_right_col = st.columns(2)

with second_left_col:
    st.markdown('<h2 class="subheader">Determine Your Cluster</h2>', unsafe_allow_html=True)
    if st.button("Cluster Me"):
        try:
            cluster_id = kmeans.predict(scaler.transform(input_df))[0]
            st.success(f"You belong to Cluster {cluster_id}.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

with second_right_col:
    st.markdown('<h2 class="subheader">Cluster Analysis Table</h2>', unsafe_allow_html=True)
    st.dataframe(cluster_analysis.head(), height=212)

# Streamlit run K-Means_App2.py
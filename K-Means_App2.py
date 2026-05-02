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
# 2. CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
   .block-container { padding-top: 1rem; }
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #333333;
    }
    .subheader {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #555555;
    }
    .scrollable-summary {
        height: 420px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
        background-color: #f9f9f9;
    }
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
st.sidebar.image(sidebar_image, width=200)
st.sidebar.header("Cluster Visualization")

numeric_features = ['id', 'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']

categorical_options = {
    'sex':     ['Female', 'Male'],
    'dataset': ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'],
    'cp':      ['asymptomatic', 'atypical angina', 'non-anginal', 'typical angina'],
    'fbs':     ['False', 'True'],
    'restecg': ['0', 'lv hypertrophy', 'normal', 'st-t abnormality'],
    'exang':   ['False', 'True'],
    'slope':   ['0', 'downsloping', 'flat', 'upsloping'],
    'thal':    ['0', 'fixed defect', 'normal', 'reversable defect'],
}

user_inputs = {}

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
st.image(main_image, use_container_width=True)

# — Row 1: PCA scatter | Cluster Summaries —
left_col, right_col = st.columns(2)

with left_col:
    st.markdown('<h2 class="subheader">Cluster Visualization</h2>', unsafe_allow_html=True)
    fig = px.scatter(
        pca_2d_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        title="Clusters Visualized with PCA",
        labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
        template='plotly'
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown('<h2 class="subheader">Cluster Summaries</h2>', unsafe_allow_html=True)
    
    if isinstance(cluster_summaries, str):
        # Single string — render as markdown directly
        with st.container(height=420):
            st.markdown(cluster_summaries)
    
    elif isinstance(cluster_summaries, dict):
        # Dictionary — render each cluster separately
        with st.container(height=420):
            for key, value in cluster_summaries.items():
                st.markdown(f"### Cluster {key}")
                st.markdown(value)
                st.divider()
    
    else:
        st.warning("No cluster summaries available.")

# — Row 2: Cluster Me button | Cluster Analysis Table —
second_left_col, second_right_col = st.columns(2)

with second_left_col:
    st.subheader("Determine Your Cluster")
    if st.button("Cluster Me"):
        try:
            cluster_id = kmeans.predict(scaler.transform(input_df))[0]
            st.success(f"You belong to Cluster {cluster_id}.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

with second_right_col:
    st.subheader("Cluster Analysis Table")
    st.dataframe(cluster_analysis.head(), height=212)

# Streamlit run K-Means_App2.py


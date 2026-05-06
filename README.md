# 🫀 K-Means Clustering for Heart Disease Analysis

An unsupervised machine learning project that applies K-Means clustering to a UCI Heart Disease dataset, with an interactive Streamlit app for real-time cluster prediction and an LLM-powered results summary using LLaMA 3 via Ollama (running locally).

> Built following a 2-part YouTube tutorial series and updated for 2026 compatibility.
> 📺 [Part 1](https://www.youtube.com/watch?v=Io5qEs56US8) | [Part 2](https://www.youtube.com/watch?v=U78Eaa8piBw)

---

## 📌 Overview

This project explores patterns in heart disease data by grouping 920 patients into clusters based on clinical features such as age, cholesterol, chest pain type, and heart disease stage. A Streamlit web app lets users input their own health metrics and discover which cluster they belong to.

---

## ✨ Features

- **Data Preprocessing** — handles missing values, one-hot encodes categorical variables, and standardises numerical features
- **Elbow Method & Dendrogram** — determines the optimal number of clusters (k=5) using inertia analysis and hierarchical clustering
- **K-Means Clustering** — partitions 920 patient records into 5 distinct clusters
- **PCA Visualisation** — reduces dimensionality to 2D for an interactive Plotly cluster scatter plot
- **LLM Results Analysis** — uses LLaMA 3 (via Ollama, running locally) to generate plain-English summaries of each cluster's medical characteristics
- **Streamlit App** — interactive dashboard with sidebar inputs, PCA plot, scrollable cluster summaries, and a "Cluster Me" prediction button

---

## 🗂️ File Structure

```
Portfolio/
   ├── K-Means_App2.py          ← Streamlit app (main file)
   ├── generate_summaries.py    ← Ollama/LLaMA 3 summary generator (run once)
   ├── kmeans_model.pkl         ← trained KMeans model
   ├── scaler.pkl               ← trained StandardScaler
   ├── feature_columns.pkl      ← saved feature column order
   ├── pca_2d.xlsx              ← PCA coordinates for scatter plot
   ├── cluster_analysis.xlsx    ← cluster statistics table
   ├── cluster_summaries.pkl    ← LLM-generated summaries (right panel)
   ├── Pic1.PNG                 ← sidebar image
   └── Pic2.PNG                 ← main banner image
```

> All files must be in the same local folder for the app to run correctly.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Pranavtamrakar/kmeans-heart-disease.git
cd kmeans-heart-disease
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama (for LLM summaries)
Download Ollama from [ollama.com](https://ollama.com), then in a terminal run:
```bash
ollama serve
```

### 4. Generate cluster summaries (first time only)
Open a second terminal and run:
```bash
ollama pull llama3
python generate_summaries.py
```
This creates `cluster_summaries.pkl` which powers the summaries panel in the app.

### 5. Launch the Streamlit app
```bash
streamlit run K-Means_App2.py
```

> **After the first setup**, you only need to run `ollama serve` and `streamlit run K-Means_App2.py`.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Google Colab | Model training and file generation |
| VS Code | Streamlit app development |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | K-Means, PCA, StandardScaler |
| Matplotlib & Seaborn | EDA and cluster visualisation |
| Plotly | Interactive PCA scatter plot in Streamlit |
| Streamlit | Web app dashboard |
| Ollama + LLaMA 3 | Local LLM-powered cluster summaries |
| Pickle / Excel | Model and data persistence |

---

## 📊 Dataset

**Heart Disease UCI** — sourced from [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

- 920 patient records across 4 datasets: Cleveland, Hungary, Switzerland, VA Long Beach
- 15 features including age, sex, chest pain type, cholesterol, resting ECG, and heart disease stage (0–4)

---

## 🔍 Cluster Summary

| Cluster | Avg Age | Avg Heart Disease Stage | Key Characteristic |
|---|---|---|---|
| 0 | 48.9 | 0.03 | Youngest group, virtually no heart disease |
| 1 | 59.0 | 1.33 | Older males, early-stage heart disease |
| 2 | 56.4 | 1.74 | Moderate disease, elevated Ca levels |
| 3 | 57.5 | 1.80 | Most severe symptoms, very low cholesterol |
| 4 | 50.4 | 0.89 | Mild disease, highest average cholesterol |

---

## 🔄 2024 → 2026 Code Updates

Some syntax changes were required to make the tutorial code compatible with current library versions:

| What changed | Old | New |
|---|---|---|
| Streamlit image display | `use_column_width=True` | `use_container_width=True` |
| Ollama response access | `response.content` | `response['message']['content']` |
| Pandas group mean | `group.mean()` | `group.mean(numeric_only=True)` |

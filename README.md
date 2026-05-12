# End-to-End KMeans Clustering Project
### Heart Disease Patient Segmentation with PCA Visualisation & Streamlit Deployment

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-KMeans-orange)
![Ollama](https://img.shields.io/badge/Ollama-Llama3-purple)

---

## 📌 Project Overview

This project applies the **IBM Data Science Lifecycle** to segment heart disease patients into distinct clusters using **KMeans Unsupervised Machine Learning**. The full pipeline covers data preparation, model development, and deployment as an interactive web application.

The project was inspired by and built upon this [YouTube tutorial series](https://www.youtube.com/watch?v=U78Eaa8piBw), with significant updates to adapt the code for 2025/2026 library versions.

---

## 🔬 Data Science Lifecycle

| Phase | What was done |
|---|---|
| **Data Preparation** | Cleaned UCI Heart Disease dataset, encoded categorical variables, scaled features with StandardScaler |
| **Data Exploration** | Visualised distributions and correlations to understand patient profiles |
| **Model Development** | Applied KMeans (K=5) and PCA dimensionality reduction |
| **Model Implementation** | Generated AI cluster summaries using Llama3 via Ollama |
| **Model Deployment** | Built and deployed an interactive Streamlit web app |

---

## 🧠 Key Modelling Decisions

### Number of Clusters (K=5)
The UCI Heart Disease dataset includes a target column with 5 heart disease stages (0–4), which guided the choice of K=5. In a project without labelled data, the **Elbow Method** would be used to find the optimal K by plotting inertia against different values of K.

### PCA Dimensionality Reduction
Principal Component Analysis (PCA) was applied to reduce the high-dimensional feature space. **2 principal components** were selected as they retained approximately **30% of the total variability** of the original features — sufficient to produce a meaningful 2D visualisation of cluster separations.

---

## 🛠️ Tool Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Google Colab | Model training environment |
| Pandas | Data manipulation |
| Scikit-learn | KMeans clustering & StandardScaler |
| Plotly | Interactive PCA scatter chart |
| Ollama + Llama3 | Local LLM for cluster summary generation |
| VS Code | App development |
| Streamlit | Web app deployment |
| Pickle / Excel | Model and data persistence |

---

## 📁 File Structure

```
📁 End-to-End-K-Means-Cluster-Project/
   ├── K-Means_App2.py            # Streamlit web app
   ├── generate_summaries.py      # Llama3 cluster summary generator
   ├── cluster_analysis.xlsx      # Cluster statistics table
   ├── cluster_summaries.pkl      # LLM-generated summaries
   ├── kmeans_model.pkl           # Trained KMeans model
   ├── scaler.pkl                 # Trained StandardScaler
   ├── pca_2d.xlsx                # PCA coordinates for scatter plot
   ├── heart_disease_uci.csv      # Original dataset
   ├── Pic1.PNG                   # Sidebar image
   ├── Pic2.PNG                   # Banner image
   └── README.md
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Step 1 — Install dependencies
```bash
pip install streamlit pandas plotly scikit-learn pillow openpyxl ollama
```

### Step 2 — Start Ollama and pull Llama3
```bash
ollama serve
ollama pull llama3
```

### Step 3 — Generate cluster summaries (run once)
```bash
python generate_summaries.py
```

### Step 4 — Launch the Streamlit app
```bash
streamlit run K-Means_App2.py
```

---

## 📊 App Features

- **Sidebar inputs** — Enter your own clinical data (age, cholesterol, blood pressure, etc.)
- **PCA Scatter Chart** — Interactive visualisation of all 5 patient clusters
- **Cluster Summaries** — AI-generated plain-English descriptions of each cluster
- **Cluster Me button** — Predicts which cluster you belong to based on your inputs
- **Cluster Analysis Table** — Statistical summary table of all clusters

---

## 📚 References

- [YouTube Tutorial Series](https://www.youtube.com/watch?v=U78Eaa8piBw)
- [IBM Data Science Lifecycle](https://public.dhe.ibm.com/software/data/sw-library/analytics/data-science-lifecycle/)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## 👤 Author

**Alberto Aguilera R.**
[GitHub](https://github.com/AlbertoAguileraR)

---

*This project was built as part of a portfolio to demonstrate end-to-end data science and ML deployment skills.*

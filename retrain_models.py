"""
retrain_models.py
-----------------
Regenerates kmeans_model.pkl, scaler.pkl, pca_2d.xlsx, and cluster_analysis.xlsx
using whatever NumPy / scikit-learn version is installed in THIS Python
environment. Run this once when you hit "ModuleNotFoundError: No module named
'numpy._core'" -- that error means the pickles were saved by NumPy 2.x and your
current NumPy is 1.x. Re-saving them under your current versions fixes it.

Usage (from the project 1 folder):
    python retrain_models.py

Requirements:
    pip install pandas scikit-learn openpyxl

The pipeline matches alberto's 34-feature schema so the existing
K-Means_App2.py keeps working without changes.
"""

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

CSV_PATH = "heart_disease_uci.csv"
N_CLUSTERS = 5
RANDOM_STATE = 42

NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# Exact 34-column order the Streamlit sidebar produces.
FEATURE_ORDER = [
    "age", "trestbps", "chol", "thalch", "oldpeak", "ca",
    "sex_Female", "sex_Male",
    "dataset_Cleveland", "dataset_Hungary", "dataset_Switzerland", "dataset_VA Long Beach",
    "cp_asymptomatic", "cp_atypical angina", "cp_non-anginal", "cp_typical angina",
    "fbs_False", "fbs_True",
    "restecg_lv hypertrophy", "restecg_normal", "restecg_st-t abnormality",
    "exang_False", "exang_True",
    "slope_downsloping", "slope_flat", "slope_upsloping",
    "thal_fixed defect", "thal_normal", "thal_reversable defect",
    "Heart Disease Stage_0", "Heart Disease Stage_1", "Heart Disease Stage_2",
    "Heart Disease Stage_3", "Heart Disease Stage_4",
]


def load_and_prepare():
    df = pd.read_csv(CSV_PATH)

    # Normalise fbs/exang to "True"/"False" strings so one-hot keys match the app
    for col in ["fbs", "exang"]:
        df[col] = df[col].astype(str).str.strip().str.capitalize()  # TRUE -> True

    # Numeric: fill missing with column median
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Categorical: fill missing with mode
    for col in CATEGORICAL:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    # "num" (0-4) -> one-hot as "Heart Disease Stage_*"
    df["Heart Disease Stage"] = df["num"].astype(int).astype(str)

    one_hot = pd.get_dummies(
        df[CATEGORICAL + ["Heart Disease Stage"]],
        prefix_sep="_",
    ).astype(int)

    X = pd.concat([df[NUMERIC].reset_index(drop=True),
                   one_hot.reset_index(drop=True)], axis=1)

    # Add any missing expected columns as zeros and reorder
    for col in FEATURE_ORDER:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_ORDER]
    return X


def main():
    X = load_and_prepare()
    print(f"Feature matrix shape: {X.shape}  (expected (920, 34))")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_coords, columns=["PCA1", "PCA2"])
    pca_df["Cluster"] = labels.astype(str)

    # Per-cluster stats table
    work = X.copy()
    work["Cluster"] = labels
    cluster_analysis = work.groupby("Cluster").mean(numeric_only=True).round(3).reset_index()

    with open("kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    pca_df.to_excel("pca_2d.xlsx", index=False)
    cluster_analysis.to_excel("cluster_analysis.xlsx", index=False)

    print("Saved:")
    print("  kmeans_model.pkl     (n_clusters =", N_CLUSTERS, ")")
    print("  scaler.pkl           (n_features =", scaler.n_features_in_, ")")
    print("  pca_2d.xlsx          (", len(pca_df), "rows )")
    print("  cluster_analysis.xlsx(", len(cluster_analysis), "rows )")
    print("Done. You can now run:  streamlit run K-Means_App2.py")


if __name__ == "__main__":
    main()

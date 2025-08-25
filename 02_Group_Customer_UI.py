import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


DEFAULT_DATA_PATH = r"C:\Programming\Prodigy Infotech\PRODIGY_ML_02\02_Clustering\Mall_Customers.csv"
ALT_DATA_PATH = "PRODIGY_ML_02\02_Clustering\Mall_Customers.csv"
PIPELINE_PATH = "customer_segmentation_pipeline.joblib"
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def find_data_path():
    if Path(DEFAULT_DATA_PATH).exists():
        return DEFAULT_DATA_PATH
    if Path(ALT_DATA_PATH).exists():
        return ALT_DATA_PATH
    raise FileNotFoundError(
        f"Could not find dataset.\nTried:\n- {DEFAULT_DATA_PATH}\n- {ALT_DATA_PATH}"
    )

def train_or_load_pipeline():
    if Path(PIPELINE_PATH).exists():
        return joblib.load(PIPELINE_PATH)

    csv_path = find_data_path()
    df = pd.read_csv(csv_path)
    X = df[FEATURES].dropna().copy()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=5, n_init=10, random_state=42)),
    ])
    pipe.fit(X)
    joblib.dump(pipe, PIPELINE_PATH)
    return pipe

def centers_in_original_units(pipe: Pipeline) -> pd.DataFrame:
    scaler: StandardScaler = pipe.named_steps["scaler"]
    kmeans: KMeans = pipe.named_steps["kmeans"]
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    df_centers = pd.DataFrame(centers_original, columns=FEATURES)
    df_centers.index.name = "Cluster"
    return df_centers.round(2)

def describe_center(row) -> str:
    return (
        f"Age≈{row['Age']:.1f}, "
        f"Income≈{row['Annual Income (k$)']:.1f}k$, "
        f"Spending≈{row['Spending Score (1-100)']:.1f}"
    )

def plot_clusters(pipe: Pipeline, user_point=None):
    """Create a 2D PCA scatter plot of clusters and optionally plot a new point."""
    csv_path = find_data_path()
    df = pd.read_csv(csv_path)
    X = df[FEATURES].dropna().copy()
    scaler: StandardScaler = pipe.named_steps["scaler"]
    kmeans: KMeans = pipe.named_steps["kmeans"]

    X_scaled = scaler.transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(kmeans.cluster_centers_)

    plt.figure(figsize=(7, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap="tab10", alpha=0.6, s=50)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X', label='Centers')

    if user_point is not None:
        user_scaled = scaler.transform(np.array([user_point]))
        user_pca = pca.transform(user_scaled)
        plt.scatter(user_pca[:, 0], user_pca[:, 1], c='black', s=150, marker='*', label='Your Input')

    plt.title("Customer Segments (PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()

    img_path = "cluster_plot.png"
    plt.savefig(img_path)
    plt.close()
    return img_path

def predict_cluster(age: float, income_k: float, spending: float):
    try:
        x = [float(age), float(income_k), float(spending)]
        x_df = pd.DataFrame([x], columns=FEATURES)
    except Exception:
        return "Invalid input", None, None, None

    pipe = predict_cluster.pipe
    scaler: StandardScaler = pipe.named_steps["scaler"]
    kmeans: KMeans = pipe.named_steps["kmeans"]

    cluster_id = int(pipe.predict([x])[0])
    centers_df = predict_cluster.centers_df
    center_desc = describe_center(centers_df.loc[cluster_id])

    x_scaled = scaler.transform(x_df)
    dists = np.linalg.norm(kmeans.cluster_centers_ - x_scaled, axis=1)
    inv = (dists.max() - dists)
    scores = (inv - inv.min()) / (inv.max() - inv.min() + 1e-12)
    scores_tbl = pd.DataFrame(
        {"Cluster": list(range(len(dists))), "Closeness (0-1)": scores.round(3)}
    ).sort_values("Cluster").reset_index(drop=True)

    text = (
        f"**Assigned Cluster:** {cluster_id}\n\n"
        f"**Segment center:** {center_desc}\n"
    )

    img_path = plot_clusters(pipe, user_point=x)

    return text, centers_df, scores_tbl, img_path

#pipeline
_pipeline = train_or_load_pipeline()
_centers_df = centers_in_original_units(_pipeline)
predict_cluster.pipe = _pipeline
predict_cluster.centers_df = _centers_df

# Gradio UI
title = "Customer Segmentation (KMeans) - Mall Customers"
desc = (
    "Enter a customer's **Age**, **Annual Income (k$)**, and **Spending Score (1-100)**. "
    "The app standardizes inputs and assigns a cluster using **KMeans (k=5)**, "
    "matching the notebook setup."
)

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}\n{desc}")

    with gr.Row():
        age = gr.Slider(15, 80, step=1, value=30, label="Age")
        income = gr.Slider(0, 150, step=1, value=60, label="Annual Income (k$)")
        spending = gr.Slider(0, 100, step=1, value=50, label="Spending Score (1-100)")

    btn = gr.Button("Predict Cluster", variant="primary")

    out_text = gr.Markdown(label="Result")
    out_centers = gr.Dataframe(
        value=_centers_df.reset_index(),
        label="Cluster Centers (original units)",
        interactive=False,
        wrap=True
    )
    out_scores = gr.Dataframe(
        headers=["Cluster", "Closeness (0-1)"],
        label="Closeness to each Cluster",
        interactive=False
    )
    out_plot = gr.Image(label="Cluster Plot (PCA Projection)")

    btn.click(
        fn=predict_cluster,
        inputs=[age, income, spending],
        outputs=[out_text, out_centers, out_scores, out_plot]
    )

if __name__ == "__main__":
    demo.launch()

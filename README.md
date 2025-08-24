# Customer Segmentation using KMeans Clustering

This project implements a **customer segmentation** system using the **KMeans clustering algorithm** on the [Mall Customers dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python).  
It provides a **Gradio web interface** where you can input a customer's age, annual income, and spending score to predict their segment and visualize the clusters.

---

## Features

- Standardizes input features using **StandardScaler**.
- Clusters customers into **5 segments** using **KMeans**.
- Displays:
  - Assigned cluster
  - Cluster center in original units
  - Closeness score to all clusters
  - 2D PCA-based scatter plot of clusters, with your input shown as a star
- Saves and loads the trained pipeline to avoid retraining every time.

---

## Requirements

- Python 3.8+
- Libraries:
  ```bash
  pip install numpy pandas scikit-learn matplotlib joblib gradio
  ```

## 📂 Project Structure

```bash

PRODIGY_ML_02/
│──── 02_Clustering/           # Dataset & preprocessing files (if any)
│ ├── 02_Group_Customer.ipynb    # Jupyter Notebook (EDA + Training)
│ ├── 02_Group_Customer_UI.py    # Gradio UI for predictions
│ ├── house_lr.joblib           # Saved ML model
│ ├── customer_segmentation_pipeline.joblib  # Saved pipeline
│ ├── README.md

```

## ⚙️ Installation

Clone the repo and install required dependencies:

```bash
git clone https://github.com/animesh33-ctrl/PRODIGY_ML_01
cd PRODIGY_ML_01
pip install -r requirements.txt
```

## How to Run

```bash

Make sure Mall_Customers.csv is in the same folder as clustering_app.py, or edit the path in the script.

Run the Gradio app:

python clustering_app.py


A local Gradio link will appear in the terminal (e.g., http://127.0.0.1:7860) where you can interact with the app.

```

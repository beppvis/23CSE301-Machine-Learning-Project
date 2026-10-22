# 23CSE301 Machine Learning Capstone (2026–27)

This repository contains the Regression, Classification, and Clustering tracks for the 23CSE301 Machine Learning capstone. The notebooks include preprocessing, feature engineering, model fitting, evaluation, and visualizations required by the course guidelines.

## Repository layout

- `notebooks/regression.ipynb` — Regression track (10 algorithms).
- `notebooks/classification.ipynb` — Classification Part A (Review 1).
- `notebooks/classification_part_b_with_pca_lda_stratified_split.ipynb` — Classification Part B (Review 2), with the ten-model comparison and additional PCA/LDA experiments.
- `notebooks/Clustering.ipynb` — Customer segmentation using K-Means and Agglomerative Hierarchical Clustering.
- `data/datatraining.txt`, `data/datatest.txt`, `data/datatest2.txt` — Occupancy Detection dataset files.
- `data/clustering/Online Retail.xlsx` — Online Retail transactions for customer segmentation.
- `app.py` — Optional local Streamlit customer-segmentation demo.
- `requirements.txt` — Python dependencies.

## Environment setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open the notebooks:

```bash
jupyter notebook
```

Run each required notebook from top to bottom before submission so saved outputs match the current code and all plots/tables are visible. Dataset paths in the notebooks are relative to the `notebooks/` directory.

## Track descriptions and result snapshots

### 1. Regression

The notebook predicts `R_Sal` using `Depthm` and `T_degC` from the CalCOFI bottle dataset. The data files are distributed via the CalCOFI/Kaggle source referenced below; confirm the required `cast.csv` and `bottle.csv` files are available locally before running a regression notebook that needs them.

Dataset source: [CalCOFI dataset on Kaggle](https://www.kaggle.com/datasets/sohier/calcofi/data) · [CalCOFI metadata](https://calcofi.org/data/oceanographic-data/bottle-database/)

Top two models from the saved `notebooks/regression.ipynb` output:

| Model | Test R² | Test RMSE | Test MAE | 5-fold CV R² (mean ± std) |
|---|---:|---:|---:|---:|
| K-Nearest Neighbors Regressor | 0.9861 | 9.4859 | 6.8998 | 0.9775 ± 0.0026 |
| Gradient Boosting Regressor | 0.9848 | 9.9199 | 7.1517 | 0.9773 ± 0.0033 |

See the notebook for the full 10-algorithm comparison and tuning details.

### 2. Classification

**Problem:** Predict room occupancy (`Occupancy` = 0 or 1) from environmental sensor readings. Part B pools the three provided files, applies the CO2 cleaning rule, and creates a reproducible stratified 80:20 split. As documented in the notebook, randomly splitting time-ordered readings may place nearby observations in both sets, so the scores should not be interpreted as a test of future-time generalization.

Saved result snapshot from `notebooks/classification_part_b_with_pca_lda_stratified_split.ipynb`:

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Bagging | 0.9942 | 0.9942 | 0.9942 | 0.9942 | 0.9994 |
| Random Forest | 0.9942 | 0.9942 | 0.9942 | 0.9942 | 0.9995 |
| Gradient Boosting | 0.9924 | 0.9925 | 0.9924 | 0.9925 | 0.9990 |
| Support Vector Machine | 0.9912 | 0.9916 | 0.9912 | 0.9913 | 0.9940 |
| K-Nearest Neighbors | 0.9912 | 0.9914 | 0.9912 | 0.9913 | 0.9987 |
| MLP | 0.9910 | 0.9913 | 0.9910 | 0.9910 | 0.9942 |
| Logistic Regression | 0.9905 | 0.9908 | 0.9905 | 0.9906 | 0.9943 |
| AdaBoost | 0.9903 | 0.9905 | 0.9903 | 0.9903 | 0.9985 |
| Decision Tree | 0.9900 | 0.9903 | 0.9900 | 0.9901 | 0.9957 |
| Gaussian Naive Bayes | 0.9678 | 0.9718 | 0.9678 | 0.9685 | 0.9936 |

Scores are a snapshot of saved notebook outputs; regenerate the table when rerunning the notebook. The notebook includes baseline-versus-tuned comparisons, confusion matrices, ROC curves, Random Forest feature importance, five-fold CV, and additional PCA/LDA pipeline experiments.

### 3. Clustering

**Problem:** Customer segmentation from the UCI Online Retail dataset using customer-level recency, frequency, and monetary/order-value features.

The notebook implements K-Means and Agglomerative Hierarchical Clustering; the elbow curve; dendrograms for Ward, complete, average and single linkage; Silhouette Score, Davies–Bouldin Index and Calinski–Harabasz Index; PCA 2D cluster visualizations for both selected algorithms; and t-SNE plots.

Saved metric snapshot:

| Algorithm | Silhouette Score (higher is better) | Davies–Bouldin Index (lower is better) | Calinski–Harabasz Index (higher is better) | Configuration |
|---|---:|---:|---:|---|
| K-Means | 0.3810 | 0.9258 | 1723.70 | k = 5 |
| Agglomerative (Ward) | 0.3512 | 0.9559 | 1433.62 | k = 5, Ward linkage |

Customer persona labels are descriptive summaries of the fitted groups, not ground-truth classes. Refer to the clustering notebook for cluster profiles and the linkage comparison.

## Optional Streamlit demo

The repository includes a local GUI for assigning an input customer to one of the five K-Means clusters. From the repository root, run:

```bash
streamlit run app.py
```

This command runs the app locally; a public deployment is not claimed by this README.

## GitHub workflow

Make meaningful commits at milestones and push the final integration branch for review. Do not submit a single bulk upload in place of meaningful commit history.

## AI assistance disclosure

AI assistance was used for code scaffolding and documentation consistency work. Team members must independently verify the code, reproduce and check the reported outputs, and author or approve the analytical interpretation in accordance with the course guidelines.

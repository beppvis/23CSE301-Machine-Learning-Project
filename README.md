# 23CSE301 Machine Learning - Capstone Project

## 1. Regression Track
- **Dataset**: [CalCOFI Bottle Database](https://www.kaggle.com/datasets/sohier/calcofi/data) ([Metadata](https://calcofi.org/data/oceanographic-data/bottle-database/))
- **Problem Statement**: Predicting the depth and salinity characteristics of ocean water.

### Setup Instructions
1. Download the dataset from Kaggle.
2. Copy `cast.csv` and `bottle.csv` into the `data/` folder.
3. Open and run `notebooks/regression.ipynb`.


## 2. Classification Track
- **Problem Statement**: Occupancy detection and room classification based on environmental sensors.
- **Notebook**: `notebooks/classification.ipynb`.


## 3. Clustering Track
- **Dataset**: [Online Retail Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Problem Statement**: Unsupervised customer segmentation using RFM (Recency, Frequency, Monetary) and transaction value metrics to identify distinct behavioral customer personas for targeted marketing.
- **Algorithms**: K-Means Clustering and Agglomerative Hierarchical Clustering.
- **Notebook**: `notebooks/Clustering.ipynb`

### Clustering Results Summary Table
| Algorithm | Silhouette Score ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Calinski-Harabasz Index ($\uparrow$) | Optimal $k$ / Linkage |
| :--- | :---: | :---: | :---: | :---: |
| **K-Means** | **0.3810** | **0.9258** | **1723.70** | $k=5$ (selected via Elbow & Silhouette curves) |
| **Agglomerative Clustering** | 0.3512 | 0.9559 | 1433.62 | $k=5$, Ward linkage (minimizes within-cluster variance) |

*Note: Visualisations include PCA 2D projections with cluster colour-coding and centroids, and non-linear t-SNE 2D manifold embeddings for all models.*

### Customer Personas Identified
1. **Champions / Loyal High-Value**: High spend, high purchase frequency, recent activity.
2. **At-Risk / Lapsed**: Low recency (inactive for >200 days), previously moderate purchasers.
3. **Occasional High-Ticket**: Low purchase frequency but very high minimum and maximum order spend.
4. **Valuable Regulars**: Consistent repeat purchases with healthy overall spend.
5. **Casual / Developing**: Lower engagement and low total expenditure.

### Customer Segmentation Web App (Interactive Demo)
The included Streamlit application provides an interactive GUI to segment new customers in real-time based on the 5-cluster K-Means model.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Enter days since the last purchase, number of purchases, total spend, and smallest/largest order values to identify the customer's segment.

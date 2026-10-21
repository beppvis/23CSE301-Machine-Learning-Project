# ML Concepts Reference

Concepts used in [`notebooks/regression_towards_chlora.ipynb`](../notebooks/regression_towards_chlora.ipynb). This document explains the machine-learning ideas; it is not dataset-specific.

---

## 1. Supervised regression

**Regression** predicts a continuous target \(y\) from features \(X\).

In this project:

- Target \(y\) = `ChlorA` (µg/L)
- Features \(X\) = oceanographic variables (depth, temperature, oxygen, time encodings, …)

We learn a function \(\hat{y} = f(X)\) on a **training** set and measure how well it works on a held-out **test** set.

---

## 2. Correlation

### What it measures

**Correlation** summarizes how two numeric variables move together.

| Sign | Meaning |
|------|---------|
| Positive | When one rises, the other tends to rise |
| Negative | When one rises, the other tends to fall |
| Near 0 | Little monotonic/linear association |

Magnitude is usually in \([-1, 1]\). Larger absolute value ? stronger association (for that measure).

### Pearson vs Kendall (what the notebook uses)

The notebook heatmap uses **Kendall’s \(\tau\)** (`corr(method="kendall")`).

| Method | Idea | Good when… |
|--------|------|------------|
| **Pearson** | Linear relationship of raw values | Roughly linear, not too many outliers |
| **Kendall** | Agreement of *ranks* / pairwise orderings | Nonlinear monotone trends, outliers, ties |

Kendall is more robust to extreme ChlorA blooms than Pearson. It answers: “Do higher ranks of feature A tend to go with higher ranks of ChlorA?” — not “Is the relationship a straight line?”

### How we use correlation in the pipeline

1. Build a **heatmap** of pairwise correlations among usable numeric columns.
2. Rank features by **\(|\mathrm{corr}(\cdot, \mathrm{ChlorA})|\)**.
3. Prefer those top correlates for imputation and modeling.

**Important:** correlation ? causation, and high correlation does not guarantee a feature is available or useful in deployment.

### Coverage filter

Only columns with enough non-missing values (e.g. ? 5% coverage) enter the heatmap. Sparse columns produce noisy, misleading correlations.

---

## 3. Skewness and `log1p`

A **right-skewed** variable has a long right tail: many small values, few huge ones (typical of ChlorA blooms).

On a raw scatter plot, rare spikes squash the rest of the points.  
**`log1p(x) = log(1 + x)`**:

- Compresses large values and expands small ones for visualization / modeling
- Stays defined at \(x = 0\) (unlike \(\log x\))

This is a transform for *interpretation* or sometimes for the target; the notebook’s models still predict raw ChlorA unless you change that.

---

## 4. Missing values and median imputation

Real data has **NaNs**. Models like linear regression / SVR / KNN need complete numeric matrices.

**Median imputation** replaces each missing entry with the column median:

- Robust to outliers (mean is pulled by blooms)
- Simple baseline

Impute using statistics from the modeling table (in a strict ML setup, fit medians on **train only** and apply to test — the notebook computes medians on the ChlorA-present frame before the station split).

---

## 5. Feature engineering

### Derived numeric features

| Feature | Meaning |
|---------|---------|
| **Line / Station** | Parsed from `Sta_ID` (CalCOFI grid position) |
| **NSM** | Number of Seconds since Midnight (time of day as a number) |
| **Day** | Day of month from the cast date |

### One-hot encoding (OHE)

Categorical integers like Month \(=\{1,\ldots,12\}\) and Day \(=\{1,\ldots,31\}\) are not ordered distances for linear models (month 12 is not “twice” month 6).

**One-hot encoding** creates binary columns, e.g. `Month_3 = 1` if March else 0.

- Linear models treat each month as its own intercept shift
- Trees can split on these bits too (sometimes less necessary)

Tradeoff: many columns ? wider matrices; polynomial expansion on OHE would explode, so poly is applied only to continuous physical features.

---

## 6. Scaling

Different features live on different scales (depth in meters vs oxygen µmol/kg vs NSM in tens of thousands).

| Scaler | What it does | Used for |
|--------|--------------|----------|
| **MinMaxScaler** | Maps each feature to roughly \([0,1]\) | Linear models, SVR, KNN |
| **StandardScaler** | Zero mean, unit variance | Polynomial + Ridge path |

Distance-based methods (**KNN**, **SVR** with RBF) and regularized linear models are sensitive to scale. Tree ensembles are **invariant** to monotone scaling of individual features, so they often skip a scaler.

---

## 7. Pipelines

A **`Pipeline`** chains steps so the same transforms fit on train and apply on test:

```text
[scaler or poly] ? [regressor]
```

This avoids leaking test statistics into preprocessing and keeps GridSearchCV clean (`model__alpha` names the step).

---

## 8. Train / test split (grouped by station)

A **random row split** can put bottles from the same station into both train and test, which often **overestimates** performance (same location “seen” already).

**Per-station split:** for each `Sta_ID`, hold out ~20% of that station’s rows for test.

- Every busy station appears in both sets
- Evaluation is closer to “new bottles at known stations”
- Still not the same as predicting a **brand-new** station never seen in training

---

## 9. Cross-validation and GridSearchCV

### Cross-validation (CV)

Split the **training** data into \(k\) folds (notebook: \(k=3\)). For each candidate hyperparameter setting:

1. Train on \(k-1\) folds  
2. Score on the held-out fold  
3. Average the scores  

This estimates generalization **without touching the final test set**.

### GridSearchCV

Tries every combination in a **parameter grid**, picks the setting with best CV score, then **refits** on the full training set.

Notebook scoring: **`neg_root_mean_squared_error`** (sklearn maximizes scores, so RMSE is negated).

---

## 10. Evaluation metrics

| Metric | Formula idea | Interpretation |
|--------|--------------|----------------|
| **RMSE** | \(\sqrt{\mathrm{mean}((y-\hat{y})^2)}\) | Typical error size; penalizes large mistakes |
| **MAE** | \(\mathrm{mean}(|y-\hat{y}|)\) | Average absolute error; more robust to outliers |
| **\(R^2\)** | \(1 - \frac{\mathrm{SS_{res}}}{\mathrm{SS_{tot}}}\) | Fraction of variance explained vs predicting the mean; 1 = perfect, 0 ? useless mean baseline |

Report metrics on the **test** set; also log **best CV RMSE** from GridSearchCV.

---

## 11. The ten regressors

### 11.1 Linear Regression

\[\hat{y} = \beta_0 + \sum_j \beta_j x_j\]

Ordinary least squares. Fast, interpretable coefficients. No explicit regularization ? can overfit / become unstable with collinear features.

**Tuned:** `fit_intercept` on/off.

### 11.2 Ridge (\(L_2\))

Same linear form, but penalty \(\alpha \sum_j \beta_j^2\).

- Shrinks coefficients toward 0  
- Keeps all features  
- Helps with multicollinearity (e.g. oxygen measures that move together)

**Tuned:** `alpha` (stronger ? more shrinkage).

### 11.3 Lasso (\(L_1\))

Penalty \(\alpha \sum_j |\beta_j|\).

- Can drive some \(\beta_j\) **exactly to 0** ? sparse / automatic feature selection  
- Useful when many features are irrelevant  

**Tuned:** `alpha`.

### 11.4 ElasticNet

Mix of \(L_1\) and \(L_2\):

- `l1_ratio` near 1 ? Lasso-like  
- near 0 ? Ridge-like  

**Tuned:** `alpha`, `l1_ratio`.

### 11.5 Polynomial Regression

Expand continuous features with **`PolynomialFeatures(degree=d)`** (interactions and powers), then fit a linear model (here **Ridge** for stability).

- Captures simple nonlinear curves  
- Feature count grows fast with degree and width ? degree searched in \(\{2,3\}\), OHE excluded from the poly step  

**Tuned:** `poly__degree`, Ridge `alpha`.

### 11.6 Decision Tree Regressor

Recursive splits: “if depth > 50 and temp < 12 ? predict …”

- Nonlinear, no scaling needed  
- Easy to overfit if `max_depth` is unlimited  

**Tuned:** `max_depth`.

### 11.7 Random Forest Regressor

**Ensemble of trees** on bootstrap samples with random feature subsets at splits. Predictions averaged.

- Strong baseline for tabular data  
- Reduces variance vs a single tree  
- Exposes **feature importances** (how much each feature reduces impurity across trees)

**Tuned:** `n_estimators` (number of trees).

### 11.8 Gradient Boosting Regressor

Trees built **sequentially**: each new tree fits the residual errors of the current ensemble.

- Often high accuracy  
- Sensitive to `learning_rate` and number of trees  
- Slower to train than RF for the same size  

**Tuned:** `learning_rate`, `n_estimators`.

### 11.9 Support Vector Regressor (SVR)

Finds a function within an ?-tube of the targets, with kernel tricks for nonlinearity.

- **RBF kernel** maps inputs to a smoother nonlinear surface  
- Needs **scaling**  
- Training cost grows badly with \(n\) ? notebook **subsamples** train rows for SVR  

**Tuned:** `C` (regularization / margin tradeoff), `kernel` (`linear` / `rbf`).

### 11.10 K-Nearest Neighbors Regressor

Predict the average target of the \(k\) closest training points in feature space.

- Nonparametric, very local  
- Needs **scaling**  
- Slow / memory-heavy on large \(n\) ? train subsampled in the notebook  

**Tuned:** `n_neighbors` (\(k\)).

---

## 12. Two feature-set experiments

Same models, two \(X\) matrices:

1. **Physical (+ time/location encodings)** — what cheap sensors / cruise metadata can give  
2. **Physical + chemical** — add nutrients / oxygen / phaeopigments that correlate with ChlorA  

Comparing \(R^2\) / RMSE between sets measures how much chemical assays buy you vs physical-only prediction.

---

## 13. Feature importance (trees)

For Random Forest, **importance** ? total impurity decrease attributed to splits on that feature (normalized).

- High importance ? the model used that feature a lot  
- Not the same as correlation, and correlated features can share / dilute importance  

---

## 14. Subsampling for expensive models

**SVR** and **KNN** on ~200k rows are costly. The notebook:

- Fits GridSearchCV on a **random subset** of training rows for those models  
- Still evaluates on the **full test set**

That is a speed/accuracy tradeoff: CV/tuning is approximate; test metrics remain comparable across models.

---

## 15. Checkpointing results

After each model, metrics are written to CSV under `model_results/`. That supports long runs (resume / inspect mid-flight) and reloading without retraining (§7.3).

---

## Quick glossary

| Term | One-liner |
|------|-----------|
| Target | Variable you predict (`ChlorA`) |
| Feature | Input column used by the model |
| Overfitting | Fits train noise; test metrics collapse |
| Regularization | Penalty that discourages large/complex weights |
| Hyperparameter | Setting chosen by search (`alpha`, `max_depth`, …), not learned like \(\beta\) |
| Leakage | Test information used during training/preprocessing |
| Ensemble | Combine many weak models (RF, GBM) |

---

*Aligned with techniques appearing in `regression_towards_chlora.ipynb`: Kendall correlation, log1p EDA, median impute, OHE, NSM, Pipelines, scalers, per-station split, GridSearchCV, and the ten regressors above.*

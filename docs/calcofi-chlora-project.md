# CalCOFI ChlorA Project Guide

Project-specific write-up for [`notebooks/regression_towards_chlora.ipynb`](../notebooks/regression_towards_chlora.ipynb): what the data is, what we are building, and why we made the choices we did.

For pure ML theory (algorithms, metrics, correlation types), see [`ml-concepts.md`](ml-concepts.md).

---

## 1. What we are trying to do (the fishing scenario)

**Goal in one sentence:** predict **chlorophyll-a (`ChlorA`)** from measurements a boat can realistically collect, so fishermen (and managers) get a cheap proxy for **where phytoplankton — and therefore fish food — is likely concentrated**.

### Why ChlorA?

Chlorophyll-a is the pigment in phytoplankton. Regions and depths with higher ChlorA are often more productive. Fish (and fisheries) track that productivity. A model that estimates ChlorA without a full lab stack can support:

- Choosing **where / how deep** to fish on a given day  
- Planning trips using **temperature, depth, location, time** (and optionally cheap chemical sensors)  
- Avoiding dependence on **expensive quality-controlled lab assays** for every cast  

### The cost constraint (why we drop “quality” fields)

CalCOFI bottles ship with many **quality codes** and **precision flags** (`*q`, `*_qual`, `*_prec`, …). Those exist because research ships run calibrated, audited instruments.

In our scenario those flags are **not a product feature**:

- The sensors / QA workflow behind them are **costly**  
- A fishing vessel will not reproduce CalCOFI’s full quality pipeline  
- Training on quality codes would leak “lab-grade trust” the end user does not have  

So in EDA and modeling we **exclude quality/precision columns** and focus on quantities that map to **physical sensors**, **location/time**, and (in a richer setup) **chemical assays** that are still more deployable than full QA metadata.

We still use **measured ChlorA as the training target** (historical research data). At inference time, the point is to **approximate ChlorA** from cheaper inputs — not to require a fluorometer + QA stack on every trip.

---

## 2. What the dataset is (CalCOFI)

**CalCOFI** (California Cooperative Oceanic Fisheries Investigations) is a long-running ocean observing program off California / Baja. Public tables used here:

| File | Grain | Role |
|------|--------|------|
| `bottle.csv` | One row per bottle / depth sample | Water properties + **ChlorA** |
| `cast.csv` | One row per cast (station occupation) | Position, time, distance from shore, cruise metadata |
| `* Field Descriptions.csv` | Column dictionaries | Human-readable units / meanings |

Sources commonly cited: [Kaggle CalCOFI](https://www.kaggle.com/datasets/sohier/calcofi/data), [CalCOFI bottle database metadata](https://calcofi.org/data/oceanographic-data/bottle-database/).

### Join key

Bottles belong to casts via **`Cst_Cnt`**. Section 3 left-joins cast fields onto every bottle so each sample gets Month, Distance, Lat/Lon, Time, etc.

Only overlapping non-key column in practice: **`Sta_ID`** (kept from bottle).

### Scale (approx.)

- ~895k bottle rows, ~36k casts  
- ChlorA present on only a **subset** of bottles (~245k) — fluorometric chlorophyll was not measured on every historic sample  
- Modeling uses **ChlorA-not-null** rows only  

---

## 3. Target and why it is hard

| Item | Detail |
|------|--------|
| Target | `ChlorA` (µg/L), fluorometric chlorophyll-a |
| Shape | Strongly **right-skewed** (many near-zero, rare blooms) |
| Implication | Raw scatter plots look crushed; EDA uses `log1p(ChlorA)` to see structure |

ChlorA is biologically noisy: depth, season, nutrients, light, and location all interact. A single linear model on physical fields alone is weak; trees + chemical features do much better (see results below).

---

## 4. End-to-end approach (mapped to the notebook)

```text
Load bottle + cast
    ? join on Cst_Cnt
    ? EDA (correlation, distance–ChlorA, seasonality)
    ? engineer Line/Station, NSM, Month/Day OHE; median-impute top correlates
    ? per-station train/test split
    ? train 10 regressors × 2 feature sets (GridSearchCV)
    ? compare physical vs physical+chemical; inspect RF importances
```

### Section map

| § | What happens | Why it matters for the scenario |
|---|--------------|----------------------------------|
| 1–2 | Load data + field dictionaries | Know which columns are sensors vs QA junk |
| 3 | Join bottle ? cast | Attach location/time/distance to every depth sample |
| 4 | EDA without quality columns | Find drivers of ChlorA fishermen might approximate |
| 5 | Impute + encode time/location | Build a model matrix from usable signals |
| 6 | Split **within each station** | Honest test: new bottles, familiar stations |
| 7 | Model zoo + two feature budgets | Physical-only ? cheap boat; +chem ? richer sensors |

---

## 5. Decisions and tradeoffs

### 5.1 Drop quality / precision / ID clutter

**Decision:** Remove `*q`, `*_qual`, `*_prec`, counters, and similar from correlation and modeling views.

**Tradeoff:** Lose information that a research QA officer would use.  
**Rationale:** Matches the fishing deployment story — those fields are expensive and unavailable.

### 5.2 Two feature budgets (physical vs physical + chemical)

**Physical-only (cheap boat story)**

- Temperature (`T_degC`)  
- Depth (`Depthm`, `R_Depth`), dynamic height (`R_DYNHT`)  
- Location: **Line** and **Station** (CalCOFI grid)  
- Time: **NSM**, one-hot **Month** and **Day**

**Physical + chemical (richer but costlier)**

- Above, plus top correlates: `Phaeop`, oxygen measures, `O2Sat`, `SiO3uM`, `NO3uM`, …

**Tradeoff:** Chemical features raise accuracy but need nutrient/oxygen sensors or lab samples. Physical-only is weaker but closer to “CTD + GPS + clock.”

This dual setup answers: *How much ChlorA signal do we lose if we refuse chemical inputs?*

### 5.3 Median imputation on top correlates

**Decision:** Fill NaNs in top ChlorA-related columns (and Line/Station/NSM) with medians.

**Tradeoff:** Invents values; can blur real missingness patterns.  
**Rationale:** Keeps sample size large; medians resist bloom outliers better than means.

### 5.4 One-hot Month and Day + NSM

**Decision:** Encode calendar seasonality and time of day explicitly.

**Tradeoff:** Many dummy columns; Day-of-month is a coarse seasonal proxy (not day-of-year).  
**Rationale:** Phytoplankton and fishing conditions vary by season and diurnally; NSM gives a continuous clock feature.

### 5.5 Per-station train/test split (not random rows)

**Decision:** Within each `Sta_ID`, ~80/20 split.

**Tradeoff:** Does **not** test “brand-new station never seen.”  
**Rationale:** Fishing often returns to known grounds; we want generalization to **new casts/bottles at known lines**, not inflated scores from leaking the same station’s near-duplicates randomly into train and test.

### 5.6 Kendall correlation for EDA

**Decision:** Heatmap uses Kendall \(\tau\).

**Tradeoff:** Slower / less familiar than Pearson.  
**Rationale:** ChlorA outliers and monotone-but-nonlinear links are common in ocean data.

### 5.7 Model zoo + GridSearchCV

**Decision:** Compare 10 standard regressors with small hyperparameter grids and 3-fold CV.

**Tradeoff:** Runtime (RF/GBM especially); SVR/KNN need **subsampling** on ~200k train rows.  
**Rationale:** Course-style coverage of linear, regularized, poly, tree, ensemble, SVR, KNN — and an empirical winner for this table.

### 5.8 Polynomial features only on continuous physical fields

**Decision:** Do not expand Month/Day dummies with `PolynomialFeatures`.

**Tradeoff:** Miss some interactions between season dummies and depth/temp inside the poly model (trees can still capture them).  
**Rationale:** Degree-2/3 on dozens of OHE columns blows up memory and noise.

### 5.9 Predict raw ChlorA (not log target)

**Decision:** Models optimize RMSE/\(R^2\) on µg/L ChlorA; log is for EDA plots.

**Tradeoff:** Errors on bloom events dominate RMSE.  
**Rationale:** Fishermen care about actual concentration scale; log-target would need careful back-transform and metrics on the original scale anyway.

---

## 6. What EDA told us (dataset-specific)

- **Phaeop, oxygen, depth, temperature, silicate, nitrate** sit among the strongest ChlorA associates — biology + light/nutrient/depth structure.  
- **Distance from shore × month:** coastal / seasonal patterns show more clearly on `log1p(ChlorA)`.  
- **Stations differ** in mean ChlorA; location features (Line/Station) are legitimate predictors for a returning fleet.

These motivate: (1) physical depth/temp/location/time as the minimum stack, (2) chemical add-ons when affordable.

---

## 7. Results snapshot (from `model_results/metrics_all.csv`)

Approximate test metrics from the trained run:

| Setup | Best model | Test \(R^2\) | Test RMSE |
|-------|------------|--------------|-----------|
| Physical only | Random Forest | ~0.78 | ~0.57 |
| Physical + chemical | Random Forest | ~0.83 | ~0.50 |

**Takeaways for the fishing story**

1. **Physical-only already beats linear models badly** — nonlinear structure (depth × season × place) matters; RF captures it.  
2. **Adding chemical features lifts \(R^2\)** (~0.78 ? ~0.83 for RF) and helps GBM even more — nutrients/oxygen carry real ChlorA signal.  
3. **Linear models** jump when chemicals are added (~0.14 ? ~0.52 \(R^2\)) but still trail trees — good for interpretability demos, not the production pick.  
4. **Practical recommendation:**  
   - Constrained boat ? deploy **RF (or GBM) on physical + time + Line/Station**  
   - If oxygen/nutrient probes exist ? add them; expect a measurable accuracy bump  

Exact numbers live in `notebooks/model_results/metrics_all.csv` and notebook §7.3.

---

## 8. What this is *not*

- Not a full fish-catch predictor (we proxy **food web productivity via ChlorA**, not landings).  
- Not a substitute for stock assessment or regulations.  
- Not trained to ignore ChlorA sensors forever in science — CalCOFI ChlorA labels are how we supervise the model.  
- Not leaving-station-out validation (future work if the product must generalize to new grid points).

---

## 9. Suggested narrative for a report / viva

1. **Problem:** Help fishermen locate productive water without research-grade QA sensors.  
2. **Proxy:** Predict ChlorA from CalCOFI history.  
3. **Data:** Join bottle chemistry/physics with cast geography/time; drop quality flags.  
4. **Method:** Engineer location/time features; impute; station-wise split; GridSearch over 10 regressors; compare cheap vs rich feature sets.  
5. **Result:** Random Forest wins; physical-only is usable; chemicals help if you can afford them.  
6. **Limitation:** Skewed target, imputation, subsampled SVR/KNN, known-station evaluation.

---

## 10. File map

| Path | Role |
|------|------|
| `notebooks/regression_towards_chlora.ipynb` | Full analysis + training |
| `notebooks/_train_all_models.py` | Standalone training script |
| `notebooks/model_results/*.csv` | Saved metrics |
| `data/cast.csv` (+ local `bottle.csv`) | CalCOFI tables |
| `docs/ml-concepts.md` | Algorithm / metric theory |

---

*Scenario framing (fisheries decision support + expensive QA sensors) is the product lens for this notebook; CalCOFI remains a research dataset used as historical ground truth for ChlorA.*

# Review 1 readiness vs Capstone Guidelines

Compared against `23CSE301_ML_26_27_Capstone_Guidelines.pdf`

**Current team framing (as of this check):**
- **Regression (final code):** `mine-lohit/notebooks/.ipynb_checkpoints/review1_energy_consumption_regression_only-checkpoint.ipynb`
- **Classification (different dataset):** `notebooks/classification.ipynb` — Occupancy Detection (`data/datatraining.txt` / `datatest*.txt`), target = `Occupancy`
- Steel `Load_Type` is **not** the classification target

---

## Review 1 scope

- Full **Regression** track (10 algorithms + metrics + tuning + CV + plots)
- **Classification Part A** (5 algorithms + Acc / weighted F1 / confusion matrices)
- Clean notebook(s) with **visible outputs** + presentable narrative
- GitHub with meaningful history

Review 2 (later): Classification Part B + Clustering — **not required yet**.

---

## Rubric scorecard (estimated)

| Section | Max | Score now | Notes |
|--|--:|--:|--|
| **A Dataset & EDA** | 4 | ~3 / 4 | Steel audit + plots coded; occupancy notebook has EDA + some written notes. Regression **observation Markdown still mostly placeholders**. |
| **B Preprocessing & FE** | 3 | ~3 / 3 | Steel: OHE + Pipeline scaler + NSM/time/load features. Occupancy: scaler pipelines + humidity-ratio note. |
| **C Regression** | 9 | ~8 / 9 **code**, ~2 / 9 **as deliverable if unrun** | Checkpoint has all 10 models incl. **SVR**, GridSearch×2, 5-fold CV, diagnostics. **Critical:** checkpoint has **0 saved outputs** (never executed). |
| **D Classification Part A** | 3 | ~2.5–3 / 3 | All 5 Part-A classifiers + Acc / weighted F1 / CM + GridSearch; executed tables present (~0.99 Acc). |
| **E Presentation** | 1 | TBD | Two-notebook story is fine if you say “regression = steel energy; classification = room occupancy”. |
| **Viva** | 5 | TBD | Need crisp dual-scenario pitch (see below). |
| **Total (code content)** | 25 | ~**16–19** if you **Run All** on the final regression notebook and fill observations; **~12–14** if you present the empty-output checkpoint as-is | |

Sibling note: `review1_energy_consumption_regression_only.ipynb` (non-checkpoint) has the same pipeline **with full outputs** (RF R²≈0.9989, SVR included). Capstone sibling **skips SVR** in the result table — do not present that one as the regression deliverable.

---

## What is already strong

### Regression (checkpoint code)
1. Dataset audit, EDA plots, IQR reported not blindly deleted  
2. Preprocessing Pipeline (OHE + StandardScaler)  
3. Feature engineering aligned with Sathishkumar et al. (2020)  
4. All **10** required regressors (incl. SVR)  
5. Metrics R² / RMSE / MAE; GridSearch on ≥2 models; 5-fold CV; pred-vs-actual / residuals / feature importance / linear coefs  
6. `random_state` for reproducibility  

### Classification Part A (`notebooks/classification.ipynb`)
1. Separate occupancy dataset (train/test files)  
2. Audit + distributions + imbalance note + CO₂ anomaly filter  
3. Logistic / KNN / GNB / Decision Tree / SVC  
4. Accuracy, weighted F1, confusion matrices, GridSearch comparison table (executed)

---

## Gaps that will hurt in Review 1

### Critical
1. **Final regression checkpoint has no outputs** — Run All before the review, then save a real path (not only `.ipynb_checkpoints/`).  
2. **Regression observation cells are still prompts** (~7 placeholders) — fill 2–4 sentences each.  
3. Do **not** present `review1_energy_consumption_capstone.ipynb` as the regression file (SVR missing from results).

### Important
4. Promote checkpoint → e.g. `mine-lohit/notebooks/review1_energy_consumption_regression_only.ipynb` after a successful run (or keep main executed file if code is intentionally the same).  
5. Checkpoint uses `SVR(kernel="linear")`; executed main used `kernel="rbf"` — pick one story and match the paper/viva.  
6. GitHub / meaningful commit history for both tracks.  
7. Viva: acknowledge **CO₂ ↔ Usage** correlation / possible leakage on the steel track.

### Nice-to-have
8. Polish classification title typo (“Preidicting”) and add a short Part-A summary markdown.  
9. One-slide / one-paragraph map: which notebook answers which rubric block.

---

## Readiness verdict

| Track | Ready to present? |
|--|--|
| Regression **code** (checkpoint) | **Yes** (~90% of C) — all 10 models + tuning + CV coded |
| Regression **deliverable** (checkpoint as-is) | **No** — **0 outputs**; must execute |
| Regression narrative | **Weak** (~20%) — placeholders |
| Classification Part A | **Mostly yes** (~85–100%) — separate occupancy notebook is executed |
| Overall Review 1 | **Close** — run regression + fill observations; classification no longer the blocker |

**Practical order before review:**  
1) Run All on the final regression notebook and save outputs under `mine-lohit/notebooks/`  
2) Fill observation Markdown  
3) Open classification notebook and confirm Part-A table + one confusion matrix for the talk  
4) Rehearse the dual scenario below  

---

## Scenario: “Why even do this?”

### Regression (steel / mine-lohit)
Predict **15-minute electricity usage (`Usage_kWh`)** for a steel plant from IoT electrical signals and schedule context, so operators can anticipate demand, cut waste, and plan tariffs — same motivation as Sathishkumar et al. (2020).

### Classification (occupancy notebook)
Predict whether a **room is occupied** from environmental sensors (temperature, humidity, light, CO₂, humidity ratio) for smart-building HVAC / lighting control — different dataset, different story; that is intentional for Review 1 Part A.

### One line for the panel
“We do **energy regression** on industrial steel IoT data, and **occupancy classification** on a building-sensor dataset, so both Review 1 tracks are complete without forcing `Load_Type` into a fake classifier.”

# Paper memory: Efficient energy consumption prediction (Sathishkumar et al., 2020)

Persistent summary of the research paper used for the `mine-lohit` capstone track.
Source PDF: `mine-lohit/Efficient energy consumption prediction model for a data analytic-enabled industry building in a sma.pdf`

---

## Citation

**Sathishkumar V E, Changsun Shin & Yongyun Cho (2020).**  
*Efficient energy consumption prediction model for a data analytic-enabled industry building in a smart city.*  
Building Research & Information. DOI: [10.1080/09613218.2020.1809983](https://doi.org/10.1080/09613218.2020.1809983)

Affiliation: Department of Information and Communication Engineering, Sunchon National University, Korea.

---

## Problem & motivation

- Smart cities need efficient **industrial energy** forecasting for cost, grid planning, and sustainability.
- Focus: **small-scale steel industry** (DAEWOO Steel Co. Ltd, Gwangyang, South Korea), not residential buildings.
- Weather variables are **not** useful here: plant is open-space with no HVAC; energy driven by production/electrical factors instead.
- Goal: predict **industry energy consumption (kWh)** from IoT / smart-meter electrical features + engineered time/load features, to support energy policy and optimization.

---

## Dataset (matches our CSV)

| Paper name | Our CSV column | Type | Unit / notes |
|---|---|---|---|
| Usage | `Usage_kWh` | Continuous (target) | kWh |
| LagRP | `Lagging_Current_Reactive.Power_kVarh` | Continuous | kVarh |
| LeadRP | `Leading_Current_Reactive_Power_kVarh` | Continuous | kVarh |
| CO2 / tCO2 | `CO2(tCO2)` | Continuous | ppm / tCO2 |
| LagPF | `Lagging_Current_Power_Factor` | Continuous | % |
| LeadPF | `Leading_Current_Power_Factor` | Continuous | % |
| NSM | `NSM` | Continuous | seconds from midnight |
| Wstatus | `WeekStatus` | Categorical | Weekday / Weekend |
| Dweek | `Day_of_week` | Categorical | Mon�Sun |
| Ltype | `Load_Type` | Categorical | Light / Medium / Maximum |

**Collection details**
- 15-minute sampling for 365 days (~1 year).
- Stored via Korea Electric Power Corporation / cloud systems.
- Paper train/test: **75% / 25%** via CARET `createDataPartition` ? train 26,281 / test 8,759; 10 variables.
- Our local file: **35,040 rows � 11 columns**, **0 missing values**.
- Target distribution: long right tail; boxplot median ~5 kWh; upper whisker ~125 kWh with outliers.

**Load-type timing windows** shift by season (esp. Nov�Feb) � Light / Medium / Maximum load periods differ (paper Table 2).

**Feature engineering in paper:** derive NSM, week status, day of week from datetime (same idea as our notebook).

---

## EDA findings (paper)

- High temporal structure: lower usage on weekends; rises ~8am�8pm.
- Kendall pairs plot: strong positive corr **Usage ? CO2 (~0.78)**; **Usage ? LagRP (~0.46)**; weaker / negative with LeadRP (~?0.11).
- Weather omitted intentionally.

---

## Feature selection (paper)

1. **Boruta** � all real predictors important; dummy random vars correctly rejected. Importance roughly ranked from CO2 down to day-of-week.
2. **RFE (RF + 10-fold CV)** � after dummy-encoding categoricals ? **18 predictors**; all useful for minimizing RMSE ? paper uses **all predictors**.

RFE rank order (approx.): tCO2, LagRP, LagPF, LeadPF, NSM, Weekend, Weekday, Saturday, Sunday, LeadRP, Tuesday, Light/Max/Medium Load, then remaining weekdays.

---

## Models compared

| Model | Role | Key hyperparameters (paper optimum) |
|---|---|---|
| GLR | Linear baseline | Intercept = TRUE (no tuning) |
| CART | Single tree | cp = 0.2 |
| SVM (RBF) | Nonlinear kernel | sigma = 0.1, cost = 13 |
| KNN | Local / distance | K = 2 |
| Random Forest | Ensemble trees | mtry = 18, ntree = 300 |
| **Cubist** | Rule-based (M5-like) | committees = 36, neighbours = 3 |

- Training used **10-fold CV** for tuning (CARET grid search, R).
- Metrics: **RMSE, MAE, CV%** (coefficient of variation). Best = lowest errors.

---

## Results (test set � paper Table 6)

| Model | Test RMSE | Test MAE | Test CV% |
|---|---:|---:|---:|
| GLR | 4.86 | 2.56 | 17.7 |
| CART | 5.69 | 2.96 | 20.7 |
| KNN | 3.61 | 1.48 | 13.1 |
| SVM | 1.97 | 1.54 | 7.2 |
| RF | 1.13 | 0.36 | 4.1 |
| **Cubist** | **0.24** | **0.055** | **0.88** |

**Takeaways from paper**
- Data is **not well explained by simple linear / shallow tree** models.
- Ensembles / kernels much better; **Cubist best**.
- RF strong but **train�test gap** (overfit risk): train RMSE 0.51 vs test 1.13.
- Cubist stays excellent on both train and test.
- No missing-value preprocessing needed � clean IoT data.

---

## Contributions claimed

1. New industrial features (Lag/Lead reactive power & power factors) rarely used in prior energy papers.
2. First use of **Cubist** in this energy-consumption setting (per authors).
3. Empirical steel-industry case for smart-city / Industry 4.0 energy analytics.

**Limitations:** steel industry only; future work could include product type, building geometry, equipment runtime, anomaly alarms.

---

## Implications for our capstone (`mine-lohit`)

- Align narrative with **smart industrial energy prediction**, not CalCOFI.
- Same target and features as paper; we can cite this paper as primary literature.
- Capstone may require **10 regressors** (broader than paper�s 6) + tuning/CV � OK to extend beyond Cubist/RF/SVM/KNN/CART/linear.
- Expect strong performance from **tree / rule / kernel** models; weak from plain linear.
- Watch **leakage**: CO2 is highly correlated with Usage � discuss whether CO2 is allowed at inference (operationally co-measured vs true �prediction before use�).
- Prefer **time-aware or stratified** splits if possible; paper used random 75/25.
- Clean data ? focus on EDA, feature engineering, model comparison, diagnostics � not imputation battles.

---

## Keywords

data mining; energy consumption; feature ranking; industrial IoT; Cubist; smart city; steel industry

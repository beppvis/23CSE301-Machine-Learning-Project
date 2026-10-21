# -*- coding: utf-8 -*-
"""Train all 10 ChlorA regressors on physical and phys+chem feature sets."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "notebooks" / "model_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

print("Loading data...", flush=True)
t0 = time.time()
bottle = pd.read_csv(DATA_DIR / "bottle.csv", encoding="iso-8859-1", low_memory=False)
cast = pd.read_csv(DATA_DIR / "cast.csv", encoding="iso-8859-1", low_memory=False)
overlap = [c for c in cast.columns if c in bottle.columns and c != "Cst_Cnt"]
df = bottle.merge(cast.drop(columns=overlap), on="Cst_Cnt", how="left")
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)


def col_containing(frame, *parts):
    for c in frame.columns:
        if all(p in c for p in parts):
            return c
    raise KeyError(parts)


OXY_UMOL = col_containing(df, "Oxy_", "mol/Kg")
R_OXY_UMOL = col_containing(df, "R_Oxy")
TOP_CORR_FEATURES = [
    "Phaeop", R_OXY_UMOL, OXY_UMOL, "O2ml_L", "Depthm", "R_Depth",
    "R_DYNHT", "O2Sat", "SiO3uM", "NO3uM", "T_degC",
]

model_df = df[df["ChlorA"].notna()].copy()
sta_parts = model_df["Sta_ID"].astype(str).str.strip().str.split(expand=True)
model_df["Line"] = pd.to_numeric(sta_parts[0], errors="coerce")
model_df["Station"] = pd.to_numeric(sta_parts[1], errors="coerce")
model_df["Date_parsed"] = pd.to_datetime(model_df["Date"], errors="coerce")
model_df["Day"] = model_df["Date_parsed"].dt.day


def time_to_nsm(t):
    if pd.isna(t):
        return np.nan
    try:
        parts = str(t).split(":")
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2]) if len(parts) > 2 else 0.0
        return h * 3600 + m * 60 + s
    except Exception:
        return np.nan


model_df["NSM"] = model_df["Time"].map(time_to_nsm)
IMPUTE_COLS = TOP_CORR_FEATURES + ["Line", "Station", "NSM"]
model_df[IMPUTE_COLS] = model_df[IMPUTE_COLS].fillna(
    model_df[IMPUTE_COLS].median(numeric_only=True)
)
model_df = model_df.dropna(subset=["Month", "Day", "Line", "Station", "Sta_ID"]).copy()
model_df["Month"] = model_df["Month"].astype(int)
model_df["Day"] = model_df["Day"].astype(int)
month_ohe = pd.get_dummies(model_df["Month"], prefix="Month")
day_ohe = pd.get_dummies(model_df["Day"], prefix="Day")
model_df = pd.concat([model_df, month_ohe, day_ohe], axis=1)
MONTH_COLS = month_ohe.columns.tolist()
DAY_COLS = day_ohe.columns.tolist()
model_df[MONTH_COLS + DAY_COLS] = model_df[MONTH_COLS + DAY_COLS].astype(float)

PHYSICAL_CONTINUOUS = ["T_degC", "Depthm", "R_Depth", "R_DYNHT", "Line", "Station", "NSM"]
CHEMICAL = ["Phaeop", R_OXY_UMOL, OXY_UMOL, "O2ml_L", "O2Sat", "SiO3uM", "NO3uM"]
physical_features = PHYSICAL_CONTINUOUS + MONTH_COLS + DAY_COLS
chemical_physical_features = physical_features + CHEMICAL


def station_train_test_split(frame, station_col="Sta_ID", test_size=0.2, random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    train_idx, test_idx = [], []
    for _, group in frame.groupby(station_col):
        idx = group.index.to_numpy()
        if len(idx) == 1:
            train_idx.append(idx[0])
            continue
        n_test = max(1, int(round(len(idx) * test_size)))
        n_test = min(n_test, len(idx) - 1)
        chosen = rng.choice(idx, size=n_test, replace=False)
        test_idx.extend(chosen.tolist())
        train_idx.extend(np.setdiff1d(idx, chosen).tolist())
    return frame.loc[train_idx].copy(), frame.loc[test_idx].copy()


train_df, test_df = station_train_test_split(model_df)
print(f"Train={len(train_df):,} Test={len(test_df):,}", flush=True)
y_train, y_test = train_df["ChlorA"], test_df["ChlorA"]


def make_models_and_params():
    """Each model is wrapped in GridSearchCV with an explicit hyperparameter grid."""
    return {
        "Linear Regression": (
            Pipeline([("scaler", MinMaxScaler()), ("model", LinearRegression())]),
            {"model__fit_intercept": [True, False]},
        ),
        "Ridge Regression": (
            Pipeline([("scaler", MinMaxScaler()), ("model", Ridge())]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "Lasso Regression": (
            Pipeline([("scaler", MinMaxScaler()), ("model", Lasso(max_iter=8000))]),
            {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
        ),
        "ElasticNet Regression": (
            Pipeline([("scaler", MinMaxScaler()), ("model", ElasticNet(max_iter=8000))]),
            {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
        ),
        "Polynomial Regression": (
            Pipeline([
                ("poly", PolynomialFeatures(include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]),
            {"poly__degree": [2, 3], "model__alpha": [1.0, 10.0, 100.0]},
        ),
        "Decision Tree Regressor": (
            Pipeline([("model", DecisionTreeRegressor(random_state=RANDOM_STATE))]),
            {"model__max_depth": [3, 5, 10, 20, None]},
        ),
        "Random Forest Regressor": (
            Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__n_estimators": [50, 100, 200]},
        ),
        "Gradient Boosting Regressor": (
            Pipeline([("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
            {"model__learning_rate": [0.01, 0.05, 0.1], "model__n_estimators": [100, 200]},
        ),
        "Support Vector Regressor (SVR)": (
            Pipeline([("scaler", MinMaxScaler()), ("model", SVR())]),
            {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"]},
        ),
        "K-Nearest Neighbors Regressor": (
            Pipeline([("scaler", MinMaxScaler()), ("model", KNeighborsRegressor(n_jobs=-1))]),
            {"model__n_neighbors": [3, 5, 7, 11]},
        ),
    }


def evaluate(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def train_model_zoo(X_train, y_train, X_test, y_test, label, svr_max_rows=8000):
    results = []
    models = make_models_and_params()
    for name, (pipe, params) in models.items():
        print(f"[{label}] Training {name} ...", flush=True)
        t1 = time.time()
        Xt, yt = X_train, y_train
        X_eval = X_test

        if name == "Polynomial Regression":
            cont_cols = [c for c in PHYSICAL_CONTINUOUS if c in X_train.columns]
            Xt, yt = X_train[cont_cols], y_train
            X_eval = X_test[cont_cols]

        if "SVR" in name and len(X_train) > svr_max_rows:
            sample_idx = y_train.sample(svr_max_rows, random_state=RANDOM_STATE).index
            Xt = X_train.loc[sample_idx]
            yt = y_train.loc[sample_idx]
            print(f"  (SVR subsampled to {svr_max_rows})", flush=True)

        # KNN: subsample train for speed at this scale
        if "K-Nearest" in name and len(Xt) > 30000:
            sample_idx = yt.sample(30000, random_state=RANDOM_STATE).index
            Xt = Xt.loc[sample_idx]
            yt = yt.loc[sample_idx]
            print("  (KNN subsampled to 30000 train rows)", flush=True)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        grid.fit(Xt, yt)
        pred = grid.best_estimator_.predict(X_eval)
        metrics = evaluate(y_test, pred)
        row = {
            "feature_set": label,
            "model": name,
            **metrics,
            "best_params": json.dumps(grid.best_params_),
            "best_cv_rmse": float(-grid.best_score_),
            "seconds": round(time.time() - t1, 1),
        }
        results.append(row)
        print(
            f"  -> R2={metrics['R2']:.4f} RMSE={metrics['RMSE']:.4f} "
            f"MAE={metrics['MAE']:.4f} ({row['seconds']}s) params={grid.best_params_}",
            flush=True,
        )
        # checkpoint after each model
        pd.DataFrame(results).to_csv(OUT_DIR / f"metrics_{label}.csv", index=False)

    return pd.DataFrame(results)


X_train_phys = train_df[physical_features]
X_test_phys = test_df[physical_features]
X_train_both = train_df[chemical_physical_features]
X_test_both = test_df[chemical_physical_features]

phys = train_model_zoo(X_train_phys, y_train, X_test_phys, y_test, "physical")
both = train_model_zoo(X_train_both, y_train, X_test_both, y_test, "phys+chem")

all_metrics = pd.concat([phys, both], ignore_index=True)
all_metrics.to_csv(OUT_DIR / "metrics_all.csv", index=False)
print("\nSaved:", OUT_DIR / "metrics_all.csv", flush=True)
print(all_metrics.sort_values(["feature_set", "R2"], ascending=[True, False]).to_string(index=False))
print("DONE", flush=True)

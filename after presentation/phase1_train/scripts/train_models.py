"""Phase 1: train Random Forest, XGBoost and LightGBM on 2002 to 2015 only.

Test years (2016 to 2024) are not read here. Scoring happens in Phase 4.

Selection rule, fixed before the run:
  The mapping model is the one with the highest mean spatial-block CV AUC on
  the training years. RF uses the Trial 7 settings (not searched again).
  XGBoost and LightGBM use the small grids below. Ties go to RF.

Writes to outputs/:
  cv_results.csv          CV AUC for every configuration of every model
  final_settings.csv      settings used for each model and the selected model
  reproduction_check.csv  RF refit compared with the Trial 7 saved model
  models/                 fitted models for Phase 4
"""

import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ---------- Paths, relative to the Phase 1 folder ----------
BASE = Path(__file__).resolve().parents[1]
TABLE = BASE / "data" / "modelling_table.csv"
REF_RF = BASE / "data" / "reference" / "rf_model_trial7.joblib"
OUT = BASE / "outputs"
MODELS = OUT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# ---------- Settings ----------
SEED = 42
N_FOLDS = 5
N_JOBS = 2  # keep the PC responsive
FEATURES = [
    "dem", "slope", "aspect_east", "aspect_north", "rivers_dist", "roads_dist",
    "settlement_dist", "rain_anomaly", "ndvi_peak", "ndmi_anomaly", "lst_anomaly",
    "lulc", "burn_count_3yr", "years_since_burn",
]
RF_SETTINGS = {  # Trial 7 final settings (phase 6a best_settings.csv)
    "n_estimators": 300,
    "min_samples_leaf": 30,
    "max_samples": 0.5,
    "max_features": 0.5,
    "max_depth": 14,
}

# ---------- Load training years only ----------
table = pd.read_csv(TABLE)
train = table[table["split"] == "train"].reset_index(drop=True)
assert train["year"].max() <= 2015, "Training table contains years after 2015"
X = train[FEATURES].to_numpy()
y = train["burned"].to_numpy()
groups = train["block_id"].to_numpy()
folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
print(f"Training rows: {len(train)}, blocks: {len(np.unique(groups))}, folds: {N_FOLDS}")


def pos_weight(yy):
    """Negative to positive ratio for class balancing."""
    return (yy == 0).sum() / (yy == 1).sum()


def build(name, params, yy):
    """Return an untrained model with the Phase 1 settings."""
    if name == "RandomForest":
        return RandomForestClassifier(
            **RF_SETTINGS, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=N_JOBS)
    if name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=300, subsample=0.7, colsample_bytree=0.7,
            min_child_weight=10, scale_pos_weight=pos_weight(yy),
            random_state=SEED, n_jobs=N_JOBS, eval_metric="auc", **params)
    return lgb.LGBMClassifier(
        n_estimators=300, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.7, min_child_samples=50, class_weight="balanced",
        random_state=SEED, n_jobs=N_JOBS, verbose=-1, **params)


GRIDS = {
    "RandomForest": [{}],
    "XGBoost": [{"max_depth": d, "learning_rate": lr}
                for d in (4, 6, 8) for lr in (0.05, 0.1)],
    "LightGBM": [{"num_leaves": nl, "learning_rate": lr}
                 for nl in (15, 31, 63) for lr in (0.05, 0.1)],
}

# ---------- Spatial-block CV on training years ----------
rows = []
best = {}
for name, grid in GRIDS.items():
    results = []
    for params in grid:
        aucs = []
        for fit, val in folds:
            m = build(name, params, y[fit])
            m.fit(X[fit], y[fit])
            aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
        mean, sd = float(np.mean(aucs)), float(np.std(aucs))
        results.append((mean, params))
        rows.append({"model": name, **params, "cv_auc_mean": round(mean, 4),
                     "cv_auc_sd": round(sd, 4)})
        print(f"{name} {params} CV AUC {mean:.4f} (sd {sd:.4f})", flush=True)
    best[name] = max(results, key=lambda r: r[0])

pd.DataFrame(rows).to_csv(OUT / "cv_results.csv", index=False)

# ---------- Selection ----------
ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0] != "RandomForest"))
selected_name = ranked[0][0]
print("Selected mapping model:", selected_name)

# ---------- Fit on all training rows ----------
settings_rows = []
for name, (cv_mean, params) in best.items():
    m = build(name, params, y)
    m.fit(X, y)
    joblib.dump(m, MODELS / f"{name}.joblib")
    settings_rows.append({
        "model": name,
        "settings": str({**(RF_SETTINGS if name == "RandomForest" else {}), **params}),
        "cv_auc_mean": round(cv_mean, 4),
        "selected": name == selected_name,
    })
pd.DataFrame(settings_rows).to_csv(OUT / "final_settings.csv", index=False)

# ---------- Reproduction check: RF refit against the Trial 7 saved model ----------
# Compared on training rows only, so no test data is used.
refit = joblib.load(MODELS / "RandomForest.joblib")
ref = joblib.load(REF_RF)
p_new = refit.predict_proba(X)[:, 1]
p_ref = ref.predict_proba(X)[:, 1]
diff = np.abs(p_new - p_ref)
check = pd.DataFrame([{
    "check": "RF refit vs Trial 7 saved model, training rows",
    "max_abs_diff": round(float(diff.max()), 6),
    "mean_abs_diff": round(float(diff.mean()), 6),
    "passed (max diff < 0.001)": bool(diff.max() < 0.001),
}])
check.to_csv(OUT / "reproduction_check.csv", index=False)
print(check.to_string(index=False))
print("done")
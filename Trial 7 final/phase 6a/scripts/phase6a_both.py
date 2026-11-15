"""Phase 6 (6a): tune and validate a Random Forest on the training years only.

Fire history variant for this run: both
  both         burn_count_3yr and years_since_burn
  burn_count   burn_count_3yr only
  years_since  years_since_burn only

Steps:
  1. Random search of 12 Random Forest settings, scored by spatial-block
     cross-validation (5 folds of 5 km blocks). The best mean AUC wins.
  2. The winner is re-scored with 300 trees, and also with forward chaining:
     train on the years before t, score year t, for t = 2009 to 2015.
  3. A final model is fitted on all training rows and saved.

The test years (2016 to 2024) are never used here. Stop and restart at any
time: finished settings are kept in cv_search.csv and skipped on the next run.

Writes to outputs/:
  cv_search.csv, best_settings.csv, validation_spatial.csv,
  validation_forward.csv, importance_impurity.csv, run_settings.csv,
  rf_model.joblib, validation.png
"""

import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, ParameterSampler

# ---------- Settings ----------
PHASE = "6a"
VARIANT = "both"

SEED = 42
N_FOLDS = 5
N_SEARCH = 12
TREES_SEARCH = 100
TREES_FINAL = 300
FIRST_FORWARD_YEAR = 2009
LAST_TRAIN_YEAR = 2015

# Use half of the cores so the computer stays usable
N_JOBS = max(1, (os.cpu_count() or 2) // 2)

# Paths, relative to this phase folder
BASE = Path(__file__).resolve().parents[1]
TABLE = BASE / "data" / "modelling_table.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

COMMON = [
    "dem", "slope", "aspect_east", "aspect_north",
    "rivers_dist", "roads_dist", "settlement_dist",
    "rain_anomaly", "ndvi_peak", "ndmi_anomaly", "lst_anomaly", "lulc",
]

FIRE_HISTORY = {
    "both": ["burn_count_3yr", "years_since_burn"],
    "burn_count": ["burn_count_3yr"],
    "years_since": ["years_since_burn"],
}

FEATURES = COMMON + FIRE_HISTORY[VARIANT]

# Settings to search: 3 x 3 x 3 x 2 = 54 combinations, 12 are tried
SPACE = {
    "max_depth": [6, 10, 14],
    "min_samples_leaf": [10, 30, 80],
    "max_features": ["sqrt", 0.3, 0.5],
    "max_samples": [0.3, 0.5],
}

# ---------- Data: training rows only ----------
table = pd.read_csv(TABLE)
train = table[table["split"] == "train"].reset_index(drop=True)
del table  # the test rows are never used in this phase

X = train[FEATURES].to_numpy()
y = train["burned"].to_numpy()
groups = train["block_id"].to_numpy()
years = train["year"].to_numpy()

print(f"Phase {PHASE}, variant '{VARIANT}': {len(train)} rows, {len(FEATURES)} predictors, {N_JOBS} cores")


def make_model(params, trees):
    """Random Forest with class weights for the 14% burned share."""
    return RandomForestClassifier(
        n_estimators=trees,
        class_weight="balanced_subsample",
        n_jobs=N_JOBS,
        random_state=SEED,
        **params,
    )


def auc_of(model, train_idx, test_idx):
    """Fit on one set of rows and return the AUC on another."""
    model.fit(X[train_idx], y[train_idx])
    return roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:, 1])


# ---------- Step 1: random search, scored by spatial-block CV ----------
configs = list(ParameterSampler(SPACE, n_iter=N_SEARCH, random_state=SEED))
folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))

search_file = OUT / "cv_search.csv"
results = pd.read_csv(search_file).to_dict("records") if search_file.exists() else []
done = {int(r["config_id"]) for r in results}

for config_id, params in enumerate(configs):
    if config_id in done:
        continue

    started = time.time()
    scores = [
        auc_of(make_model(params, TREES_SEARCH), fit_idx, val_idx)
        for fit_idx, val_idx in folds
    ]

    results.append({
        "config_id": config_id,
        **params,
        "cv_auc_mean": round(float(np.mean(scores)), 4),
        "cv_auc_sd": round(float(np.std(scores)), 4),
        "seconds": round(time.time() - started),
    })

    # Save after every setting, so a stopped run can resume
    pd.DataFrame(results).to_csv(search_file, index=False)
    print(f"Setting {config_id + 1} of {N_SEARCH}: AUC {np.mean(scores):.4f} "
          f"({round(time.time() - started)} s)")

search = pd.DataFrame(results).sort_values("cv_auc_mean", ascending=False)
best_id = int(search.iloc[0]["config_id"])
best = configs[best_id]

pd.DataFrame([{"config_id": best_id, **best}]).to_csv(OUT / "best_settings.csv", index=False)
print("Best settings:", best)

# ---------- Step 2a: spatial-block CV of the winner, 300 trees ----------
spatial = []

for k, (fit_idx, val_idx) in enumerate(folds, start=1):
    spatial.append({
        "fold": k,
        "blocks_held_out": len(np.unique(groups[val_idx])),
        "auc": round(auc_of(make_model(best, TREES_FINAL), fit_idx, val_idx), 4),
    })

spatial = pd.DataFrame(spatial)
spatial.to_csv(OUT / "validation_spatial.csv", index=False)

# ---------- Step 2b: forward chaining, training years only ----------
forward = []

for year in range(FIRST_FORWARD_YEAR, LAST_TRAIN_YEAR + 1):
    fit_idx = np.where(years < year)[0]
    val_idx = np.where(years == year)[0]

    forward.append({
        "validation_year": year,
        "train_years": f"{years.min()} to {year - 1}",
        "burned_share_pct": round(100 * y[val_idx].mean(), 1),
        "auc": round(auc_of(make_model(best, TREES_FINAL), fit_idx, val_idx), 4),
    })

forward = pd.DataFrame(forward)
forward.to_csv(OUT / "validation_forward.csv", index=False)

# ---------- Step 3: final model on all training rows ----------
final = make_model(best, TREES_FINAL).fit(X, y)
joblib.dump(final, OUT / "rf_model.joblib", compress=3)

pd.DataFrame({
    "predictor": FEATURES,
    "importance": final.feature_importances_.round(4),
}).sort_values("importance", ascending=False).to_csv(OUT / "importance_impurity.csv", index=False)

pd.DataFrame([
    {"setting": "phase", "value": PHASE},
    {"setting": "fire history variant", "value": VARIANT},
    {"setting": "predictors", "value": ", ".join(FEATURES)},
    {"setting": "training years", "value": f"{years.min()} to {years.max()}"},
    {"setting": "seed", "value": SEED},
    {"setting": "settings tried", "value": N_SEARCH},
    {"setting": "trees (search, final)", "value": f"{TREES_SEARCH}, {TREES_FINAL}"},
]).to_csv(OUT / "run_settings.csv", index=False)

# ---------- Figure ----------
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

left.bar(spatial["fold"].astype(str), spatial["auc"], color="#2b6cb0")
left.axhline(spatial["auc"].mean(), color="black", linestyle="-", label=f"Mean {spatial['auc'].mean():.3f}")
left.set_title("Spatial-block cross-validation\n(unseen 5 km blocks)")
left.set_xlabel("Fold")
left.set_ylabel("AUC")

right.bar(forward["validation_year"].astype(str), forward["auc"], color="#dd6b20")
right.axhline(forward["auc"].mean(), color="black", linestyle="-", label=f"Mean {forward['auc'].mean():.3f}")
right.set_title("Forward chaining\n(train on earlier years, score the next)")
right.set_xlabel("Year scored")

for ax in (left, right):
    ax.axhline(0.5, color="grey", linestyle="--", label="Chance (0.5)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper right", frameon=False)

fig.suptitle(f"Phase {PHASE}: Random Forest, fire history = {VARIANT} (training years only)")
fig.tight_layout()
fig.savefig(OUT / "validation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Summary ----------
print("\nSpatial-block CV AUC: mean", round(spatial["auc"].mean(), 4),
      "sd", round(spatial["auc"].std(), 4))
print("Forward chaining AUC: mean", round(forward["auc"].mean(), 4))
print(forward[["validation_year", "burned_share_pct", "auc"]].to_string(index=False))
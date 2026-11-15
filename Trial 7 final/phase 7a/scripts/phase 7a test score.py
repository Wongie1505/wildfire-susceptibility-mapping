"""Phase 7a: score the 6a model once on the test years, then explain it.

This script is a one-shot. It refuses to run again once test_results.csv
exists, so the test years cannot be scored twice by accident.

Writes to outputs/:
  test_results.csv          AUC, top 20% capture and burned share, by year and overall
  test_predictions.csv      score for every test cell and season
  importance_grouped.csv    drop in AUC when each predictor (or group) is shuffled
  test_results.png          AUC by test year
  importance_grouped.png    predictor importance
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------- Settings ----------
SEED = 42
N_REPEATS = 5
TRIAL6_TEST_AUC = 0.571  # Trial 6 Random Forest, shown for reference only
N_JOBS = max(1, (os.cpu_count() or 2) // 2)

# Paths, relative to the Phase 7a folder
BASE = Path(__file__).resolve().parents[1]
TABLE = BASE / "data" / "modelling_table.csv"
MODEL = BASE / "data" / "rf_model.joblib"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

RESULTS_FILE = OUT / "test_results.csv"

if RESULTS_FILE.exists():
    raise RuntimeError(
        "The test years were already scored (test_results.csv exists). "
        "Scoring them again would be a second look. Delete the file only if "
        "you accept that and will say so in the dissertation."
    )

# Same order as the 6a model was trained with
COMMON = [
    "dem", "slope", "aspect_east", "aspect_north",
    "rivers_dist", "roads_dist", "settlement_dist",
    "rain_anomaly", "ndvi_peak", "ndmi_anomaly", "lst_anomaly", "lulc",
]
FIRE = ["burn_count_3yr", "years_since_burn"]
FEATURES = COMMON + FIRE

LABELS = {
    "dem": "Elevation",
    "slope": "Slope",
    "aspect_east": "Aspect (eastness)",
    "aspect_north": "Aspect (northness)",
    "rivers_dist": "Distance to rivers",
    "roads_dist": "Distance to roads",
    "settlement_dist": "Distance to settlements",
    "rain_anomaly": "Rainfall anomaly",
    "ndvi_peak": "NDVI peak",
    "ndmi_anomaly": "NDMI anomaly",
    "lst_anomaly": "LST anomaly",
    "lulc": "Land cover",
    "fire_history": "Fire history (both predictors)",
}

# ---------- Load the test rows and the saved model ----------
table = pd.read_csv(TABLE)
test = table[table["split"] == "test"].reset_index(drop=True)
del table  # training rows are not needed here

model = joblib.load(MODEL)
model.n_jobs = N_JOBS

if model.n_features_in_ != len(FEATURES):
    raise ValueError(f"Model expects {model.n_features_in_} predictors, found {len(FEATURES)}.")

X = test[FEATURES].to_numpy()
y = test["burned"].to_numpy()

# ---------- Score the test years, once ----------
score = model.predict_proba(X)[:, 1]


def top_capture(truth, scores, share=0.20):
    """Share of burned cells that fall in the highest-scored 20% of cells."""
    n_top = int(round(len(scores) * share))
    top = np.argsort(scores)[::-1][:n_top]
    return 100 * truth[top].sum() / truth.sum()


rows = []

for year, part in test.groupby("year"):
    idx = part.index.to_numpy()

    rows.append({
        "year": str(year),
        "cells": len(idx),
        "burned_pct": round(100 * y[idx].mean(), 1),
        "auc": round(roc_auc_score(y[idx], score[idx]), 4),
        "top20_capture_pct": round(top_capture(y[idx], score[idx]), 1),
    })

baseline_auc = roc_auc_score(y, score)

rows.append({
    "year": "ALL 2016-2024",
    "cells": len(y),
    "burned_pct": round(100 * y.mean(), 1),
    "auc": round(baseline_auc, 4),
    "top20_capture_pct": round(np.mean([r["top20_capture_pct"] for r in rows]), 1),
})

results = pd.DataFrame(rows)
results.to_csv(RESULTS_FILE, index=False)

pd.DataFrame({
    "cell_id": test["cell_id"],
    "year": test["year"],
    "burned": y,
    "score": score.round(5),
}).to_csv(OUT / "test_predictions.csv", index=False)

print(results.to_string(index=False))
print(f"\nTest AUC (all test years pooled): {baseline_auc:.4f}  |  Trial 6: {TRIAL6_TEST_AUC}")

# ---------- Grouped permutation importance (interpretation only) ----------
# Shuffle one predictor, or the two fire history predictors together, and see
# how far the AUC falls. The model is not changed by this step.
rng = np.random.RandomState(SEED)

groups = {name: [name] for name in COMMON}
groups["fire_history"] = FIRE

importance = []

for name, cols in groups.items():
    col_idx = [FEATURES.index(c) for c in cols]
    drops = []

    for _ in range(N_REPEATS):
        order = rng.permutation(len(X))
        shuffled = X.copy()
        shuffled[:, col_idx] = X[np.ix_(order, col_idx)]
        drops.append(baseline_auc - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))

    importance.append({
        "predictor": name,
        "auc_drop_mean": round(float(np.mean(drops)), 4),
        "auc_drop_sd": round(float(np.std(drops)), 4),
    })

importance = pd.DataFrame(importance).sort_values("auc_drop_mean", ascending=False)
importance.to_csv(OUT / "importance_grouped.csv", index=False)

print("\nLargest AUC drops when shuffled:")
print(importance.head(5).to_string(index=False))

# ---------- Figure 1: AUC by test year ----------
yearly = results.iloc[:-1]

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(yearly["year"], yearly["auc"], color="#dd6b20")
ax.axhline(baseline_auc, color="black", label=f"All test years {baseline_auc:.3f}")
ax.axhline(TRIAL6_TEST_AUC, color="#2b6cb0", linestyle=":", label=f"Trial 6 test AUC {TRIAL6_TEST_AUC}")
ax.axhline(0.5, color="grey", linestyle="--", label="Chance (0.5)")
ax.set_ylim(0.4, 0.9)
ax.set_xlabel("Test year")
ax.set_ylabel("AUC")
ax.set_title("Phase 7a: Random Forest (6a) scored once on the test years")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "test_results.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 2: grouped importance ----------
ranked = importance.sort_values("auc_drop_mean")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(
    [LABELS[p] for p in ranked["predictor"]],
    ranked["auc_drop_mean"],
    xerr=ranked["auc_drop_sd"],
    color="#2b6cb0",
)
ax.set_xlabel("Drop in test AUC when the predictor is shuffled")
ax.set_title("Predictor importance (fire history permuted as one group)")
fig.tight_layout()
fig.savefig(OUT / "importance_grouped.png", dpi=300, bbox_inches="tight")
plt.close(fig)
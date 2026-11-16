"""Phase 10a: descriptive checks on the 6a model, scored on the test years.

The model is not changed and nothing here is used to choose settings. The test
scores come from Phase 7a (test_predictions.csv). The model file is only used
to ask what the score does when one predictor is moved (ALE plots).

Writes to outputs/:
  bootstrap_auc.csv          test AUC and top 20% capture with block bootstrap intervals
  capture_curve.csv          share of burned cells caught against share of area flagged
  capture_curve.png          the same, pooled and by year
  threshold_sensitivity.csv  area flagged, burned cells caught, false alarms, per threshold
  calibration_bins.csv       mean score and observed burned share per score bin
  calibration_summary.csv    Brier score, baseline Brier, calibration slope and intercept
  calibration.png            reliability plot
  ale_<predictor>.csv/png    accumulated local effect for the main predictors
  lulc_summary.csv           mean score and burned share per land cover class
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

SEED = 42
N_BOOT = 1000
N_ALE_BINS = 20
N_JOBS = max(1, (os.cpu_count() or 2) // 2)
PHASE7A_AUC = 0.6029  # pooled test AUC reported in Phase 7a, used as a check

BASE = Path(__file__).resolve().parents[1]
TABLE = BASE / "data" / "modelling_table.csv"
PREDS = BASE / "data" / "test_predictions.csv"
MODEL = BASE / "data" / "rf_model.joblib"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "dem", "slope", "aspect_east", "aspect_north",
    "rivers_dist", "roads_dist", "settlement_dist",
    "rain_anomaly", "ndvi_peak", "ndmi_anomaly", "lst_anomaly", "lulc",
    "burn_count_3yr", "years_since_burn",
]

# Predictors that carried weight in Phase 7a, plus elevation for reference.
# Land cover is a class code, so it gets a table instead of an ALE curve.
ALE_PREDICTORS = {
    "burn_count_3yr": "Burns in previous 3 seasons",
    "years_since_burn": "Years since last burn",
    "rivers_dist": "Distance to rivers (m)",
    "roads_dist": "Distance to roads (m)",
    "settlement_dist": "Distance to settlements (m)",
    "dem": "Elevation (m)",
}

# Load the table, keep the test rows and attach the Phase 7a scores
table = pd.read_csv(TABLE)
train_prevalence = table.loc[table["split"] == "train", "burned"].mean()
test = table[table["split"] == "test"].reset_index(drop=True)
del table

preds = pd.read_csv(PREDS)[["cell_id", "year", "score"]]
test = test.merge(preds, on=["cell_id", "year"], how="left")
if test["score"].isna().any():
    raise ValueError("Some test rows have no score in test_predictions.csv.")

y = test["burned"].to_numpy()
score = test["score"].to_numpy()
year = test["year"].to_numpy()
block = test["block_id"].to_numpy()

pooled_auc = roc_auc_score(y, score)
print(f"Pooled test AUC {pooled_auc:.4f} (Phase 7a reported {PHASE7A_AUC})")
if abs(pooled_auc - PHASE7A_AUC) > 0.001:
    print("Warning: this does not match Phase 7a. Check that the files are the same ones.")


def top_capture(truth, scores, share=0.20):
    """Share of burned cells that fall in the highest-scored share of cells."""
    n_top = int(round(len(scores) * share))
    order = np.argsort(-scores, kind="stable")[:n_top]
    return truth[order].sum() / truth.sum()


# 1. Bootstrap interval for the test AUC
# Whole 5 km blocks are resampled, with all their seasons, because cells in a
# block are alike and cells in a season share the same weather.
rng = np.random.default_rng(SEED)
block_ids = np.unique(block)
rows_by_block = {b: np.where(block == b)[0] for b in block_ids}

boot_auc, boot_cap = [], []
for _ in range(N_BOOT):
    chosen = rng.choice(block_ids, size=len(block_ids), replace=True)
    idx = np.concatenate([rows_by_block[b] for b in chosen])
    if y[idx].min() == y[idx].max():
        continue
    boot_auc.append(roc_auc_score(y[idx], score[idx]))
    boot_cap.append(top_capture(y[idx], score[idx]))

yearly_auc = [roc_auc_score(y[year == t], score[year == t]) for t in np.unique(year)]
boot_table = pd.DataFrame({
    "measure": ["Pooled AUC", "Top 20% capture (pooled)", "Mean of yearly AUC"],
    "estimate": [pooled_auc, top_capture(y, score), np.mean(yearly_auc)],
    "lower_95": [np.percentile(boot_auc, 2.5), np.percentile(boot_cap, 2.5), np.nan],
    "upper_95": [np.percentile(boot_auc, 97.5), np.percentile(boot_cap, 97.5), np.nan],
    "resamples": [len(boot_auc), len(boot_cap), np.nan],
})
boot_table.to_csv(OUT / "bootstrap_auc.csv", index=False)
print(boot_table.round(4).to_string(index=False))


# 2. Capture against area, pooled and by year
def gain_curve(truth, scores):
    order = np.argsort(-scores, kind="stable")
    caught = np.cumsum(truth[order]) / truth.sum()
    area = np.arange(1, len(scores) + 1) / len(scores)
    return area, caught


grid = np.linspace(0, 1, 101)
curve = {"area_share": grid}
area, caught = gain_curve(y, score)
curve["pooled"] = np.interp(grid, area, caught)
for t in np.unique(year):
    a, c = gain_curve(y[year == t], score[year == t])
    curve[str(t)] = np.interp(grid, a, c)
pd.DataFrame(curve).to_csv(OUT / "capture_curve.csv", index=False)

fig, ax = plt.subplots(figsize=(5.5, 5))
for t in np.unique(year):
    ax.plot(grid, curve[str(t)], color="grey", alpha=0.35, lw=0.8)
ax.plot(grid, curve["pooled"], color="darkred", lw=2, label="All test years pooled")
ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="Chance")
ax.plot([], [], color="grey", lw=0.8, label="Single years")
ax.set_xlabel("Share of forest area flagged (highest scores first)")
ax.set_ylabel("Share of burned cells caught")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "capture_curve.png", dpi=300)
plt.close(fig)

# 3. Threshold sensitivity
rows = []
n_burned, n_unburned = y.sum(), (1 - y).sum()


def add_rule(rule, flagged):
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    rows.append({
        "rule": rule,
        "area_flagged_pct": 100 * flagged.mean(),
        "burned_caught_pct": 100 * tp / n_burned,
        "burned_missed_pct": 100 * (n_burned - tp) / n_burned,
        "false_alarm_pct_of_unburned": 100 * fp / n_unburned,
        "precision_pct": 100 * tp / max(tp + fp, 1),
        "lift": (tp / max(tp + fp, 1)) / y.mean(),
    })


for cut in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    add_rule(f"score >= {cut:.2f}", score >= cut)
for share in [0.1, 0.2, 0.3, 0.4, 0.5]:
    cut = np.quantile(score, 1 - share)
    add_rule(f"top {int(share * 100)}% of cells", score >= cut)
pd.DataFrame(rows).round(2).to_csv(OUT / "threshold_sensitivity.csv", index=False)

# 4. Calibration on the test years
bins = pd.qcut(score, 10, duplicates="drop")
cal = (
    pd.DataFrame({"score": score, "burned": y, "bin": bins})
    .groupby("bin", observed=True)
    .agg(cells=("burned", "size"), mean_score=("score", "mean"), observed_share=("burned", "mean"))
    .reset_index(drop=True)
)
cal.to_csv(OUT / "calibration_bins.csv", index=False)

clipped = np.clip(score, 1e-4, 1 - 1e-4)
logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
brier = brier_score_loss(y, score)
brier_base = brier_score_loss(y, np.full(len(y), train_prevalence))
pd.DataFrame({
    "measure": [
        "Brier score of the model",
        "Brier score of a constant at the training burned share",
        "Brier skill score (1 is perfect, 0 is no better than the constant)",
        "Calibration slope (1 is ideal)",
        "Calibration intercept (0 is ideal)",
        "Mean score on test rows",
        "Observed burned share on test rows",
    ],
    "value": [brier, brier_base, 1 - brier / brier_base, fit.coef_[0][0],
              fit.intercept_[0], score.mean(), y.mean()],
}).round(4).to_csv(OUT / "calibration_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="Perfect calibration")
ax.plot(cal["mean_score"], cal["observed_share"], marker="o", color="darkred", label="Test years")
ax.set_xlabel("Mean model score in bin")
ax.set_ylabel("Observed burned share in bin")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "calibration.png", dpi=300)
plt.close(fig)

# 5. Accumulated local effects on the test rows
# ALE moves one predictor across each bin, keeps the others as they are, and
# sums the change in score. It copes with correlated predictors better than a
# partial dependence plot. The values are changes in the model score, not in
# probability.
model = joblib.load(MODEL)
model.n_jobs = N_JOBS
X = test[FEATURES].to_numpy()


def ale(feature, n_bins):
    j = FEATURES.index(feature)
    x = X[:, j]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return None
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    effect = np.zeros(len(edges) - 1)
    count = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        rows_k = np.where(which == k)[0]
        count[k] = len(rows_k)
        if len(rows_k) == 0:
            continue
        lo, hi = X[rows_k].copy(), X[rows_k].copy()
        lo[:, j], hi[:, j] = edges[k], edges[k + 1]
        effect[k] = (model.predict_proba(hi)[:, 1] - model.predict_proba(lo)[:, 1]).mean()
    curve_at_edges = np.concatenate([[0], np.cumsum(effect)])
    mid = (curve_at_edges[:-1] + curve_at_edges[1:]) / 2
    curve_at_edges -= (mid * count).sum() / count.sum()
    return pd.DataFrame({"value": edges, "ale": curve_at_edges})


for feature, label in ALE_PREDICTORS.items():
    result = ale(feature, N_ALE_BINS)
    if result is None:
        print(f"Skipped {feature}: no variation in the test rows")
        continue
    result.to_csv(OUT / f"ale_{feature}.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(result["value"], result["ale"], color="darkred", marker="o", ms=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel("Change in model score")
    fig.tight_layout()
    fig.savefig(OUT / f"ale_{feature}.png", dpi=300)
    plt.close(fig)
    print(f"ALE done: {feature}")

lulc = (
    test.groupby("lulc")
    .agg(cell_seasons=("burned", "size"), mean_score=("score", "mean"), burned_share=("burned", "mean"))
    .reset_index()
)
lulc.round(4).to_csv(OUT / "lulc_summary.csv", index=False)
print("Phase 10a finished.")
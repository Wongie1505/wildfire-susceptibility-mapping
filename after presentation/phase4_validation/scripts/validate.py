"""Phase 4: score the four Phase 1 models on the 2016 to 2024 test years, once.

Two references:
  MCD64A1 (primary, 2016 to 2024), binary burned label from the modelling table.
  Sentinel-2 (independent, 2018 to 2024), usable rows only, from Phase 3.

Headline metric for Sentinel-2: soft AUC on the continuous burned fraction.
  Each cell acts as a burned item with weight f and an unburned item with
  weight 1 - f. Ties count as 0.5. With binary labels this equals standard AUC.

Binary metrics for Sentinel-2 use the primary cut-off 0.10 (kappa, top-20%
capture). Cut-offs 0.05 and 0.25 are sensitivity checks, reported in full.

Block bootstrap: resample the 92 test blocks, 1000 times, default_rng(42).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODELS = DATA / "models"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "dem", "slope", "aspect_east", "aspect_north", "rivers_dist", "roads_dist",
    "settlement_dist", "rain_anomaly", "ndvi_peak", "ndmi_anomaly", "lst_anomaly",
    "lulc", "burn_count_3yr", "years_since_burn",
]
MODEL_FILES = {
    "RandomForest": "RandomForest.joblib",
    "XGBoost_depth8": "XGBoost.joblib",
    "LightGBM_1SE_31leaves": "LightGBM_1SE_31leaves.joblib",
    "LightGBM_min_error_63leaves": "LightGBM_min_error_63leaves.joblib",
}
CUTS = {"primary_0.10": 0.10, "low_0.05": 0.05, "high_0.25": 0.25}
N_BOOT = 1000
SEED = 42

table = pd.read_csv(DATA / "modelling_table.csv")
test = table[table["split"] == "test"].reset_index(drop=True)
assert test["year"].between(2016, 2024).all()

labels = pd.read_csv(DATA / "s2_labels.csv")
test = test.merge(
    labels[["cell_id", "year", "burned_fraction", "usable"]],
    on=["cell_id", "year"], how="left")
test["s2_frac"] = test["burned_fraction"]
test["s2_ok"] = test["usable"].fillna(False).astype(bool) & test["s2_frac"].notna()

models = {k: joblib.load(MODELS / v) for k, v in MODEL_FILES.items()}
scores = {k: m.predict_proba(test[FEATURES].to_numpy())[:, 1] for k, m in models.items()}

blocks = test["block_id"].to_numpy()
block_ids = np.unique(blocks)
block_rows = {b: np.where(blocks == b)[0] for b in block_ids}


def soft_auc(score, frac):
    """AUC with fractional labels: burned weight f, unburned weight 1 - f."""
    w_pos, w_neg = frac, 1 - frac
    order = np.argsort(score)
    s = score[order]
    wp = w_pos[order]
    wn = w_neg[order]
    uniq, inv = np.unique(s, return_inverse=True)
    neg_by_val = np.bincount(inv, weights=wn, minlength=len(uniq))
    pos_by_val = np.bincount(inv, weights=wp, minlength=len(uniq))
    neg_below = np.concatenate([[0.0], np.cumsum(neg_by_val)[:-1]])
    total = np.sum(pos_by_val * (neg_below + 0.5 * neg_by_val))
    denom = np.sum(wp) * np.sum(wn)
    return total / denom if denom > 0 else np.nan


def top20_capture(score, frac):
    """Share of burned weight inside the top 20% of cells by score."""
    n = int(round(len(score) * 0.2))
    top = np.argsort(score)[::-1][:n]
    total = np.sum(frac)
    return 100 * np.sum(frac[top]) / total if total > 0 else np.nan


def block_ci(metric, y, s, mask=None):
    """95% block-bootstrap interval for a metric on the rows in mask."""
    rng = np.random.default_rng(SEED)
    keep = np.ones(len(y), bool) if mask is None else mask
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(block_ids), size=len(block_ids), replace=True)
        idx = np.concatenate([block_rows[block_ids[i]] for i in pick])
        idx = idx[keep[idx]]
        if len(idx) == 0:
            continue
        v = metric(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


y_mcd = test["burned"].to_numpy()
rows = []
for name, s in scores.items():
    # MCD64A1 primary, 2016 to 2024
    auc = roc_auc_score(y_mcd, s)
    lo, hi = block_ci(lambda a, b: roc_auc_score(a, b), y_mcd, s)
    rows.append({"model": name, "reference": "MCD64A1", "years": "2016-2024",
                 "metric": "AUC", "value": round(auc, 4),
                 "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

    # Sentinel-2, usable rows, 2018 to 2024
    m2 = test["s2_ok"].to_numpy()
    frac = test["s2_frac"].fillna(0).to_numpy()
    soft = soft_auc(s[m2], frac[m2])
    lo, hi = block_ci(lambda a, b: soft_auc(b, a), frac, s, m2)
    rows.append({"model": name, "reference": "Sentinel-2", "years": "2018-2024",
                 "metric": "soft AUC (headline)", "value": round(soft, 4),
                 "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

    cap = top20_capture(s[m2], frac[m2])
    rows.append({"model": name, "reference": "Sentinel-2", "years": "2018-2024",
                 "metric": "top-20% burned capture %", "value": round(cap, 2),
                 "ci_low": np.nan, "ci_high": np.nan})

    for label, cut in CUTS.items():
        yb = (frac[m2] >= cut).astype(int)
        # Kappa needs the predicted class: top 20% of cells flagged as burned
        n = int(round(m2.sum() * 0.2))
        pred = np.zeros(m2.sum(), int)
        pred[np.argsort(s[m2])[::-1][:n]] = 1
        kappa = cohen_kappa_score(yb, pred)
        rows.append({"model": name, "reference": "Sentinel-2", "years": "2018-2024",
                     "metric": f"kappa, top-20% flag vs label {label}",
                     "value": round(kappa, 4), "ci_low": np.nan, "ci_high": np.nan})

res = pd.DataFrame(rows)
res.to_csv(OUT / "validation_results.csv", index=False)
print(res.to_string(index=False))
print("done")
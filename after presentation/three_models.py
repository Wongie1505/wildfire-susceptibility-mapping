"""Compare RF, XGBoost and LightGBM on the Trial 7 design.

Training 2002 to 2015, tuning with spatial-block CV on training years only,
one scoring of the test years 2016 to 2024. Scored on MCD64A1 labels (block
bootstrap, as in the dissertation) and on FIRMS detections (per season).
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

T7 = "/home/claude/wrepo/Trial 7 final"
TABLE = f"{T7}/phase 7a/data/modelling_table.csv"
RF_MODEL = f"{T7}/phase 7a/data/rf_model.joblib"
FIRMS = f"{T7}/phase 9a/data/fire_detections_filtered.csv"
OUT = "/home/claude/runs"

FEATURES = [
    "dem", "slope", "aspect_east", "aspect_north", "rivers_dist", "roads_dist",
    "settlement_dist", "rain_anomaly", "ndvi_peak", "ndmi_anomaly", "lst_anomaly",
    "lulc", "burn_count_3yr", "years_since_burn",
]
SEED = 42
N_BOOT = 1000
CELL = 500.0

table = pd.read_csv(TABLE)
train = table[table.split == "train"].reset_index(drop=True)
test = table[table.split == "test"].reset_index(drop=True)
X_tr, y_tr = train[FEATURES].to_numpy(), train.burned.to_numpy()
X_te, y_te = test[FEATURES].to_numpy(), test.burned.to_numpy()
groups = train.block_id.to_numpy()
cv = list(GroupKFold(n_splits=5).split(X_tr, y_tr, groups))


def pos_weight(y):
    """Negative to positive ratio, used to balance the classes."""
    return (y == 0).sum() / (y == 1).sum()


def make(name, params, y):
    """Build a fresh model with balanced class weights."""
    if name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=300, subsample=0.7, colsample_bytree=0.7,
            min_child_weight=10, scale_pos_weight=pos_weight(y),
            random_state=SEED, n_jobs=-1, eval_metric="auc", **params)
    return lgb.LGBMClassifier(
        n_estimators=300, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.7, min_child_samples=50, class_weight="balanced",
        random_state=SEED, n_jobs=-1, verbose=-1, **params)


GRIDS = {
    "XGBoost": [{"max_depth": d, "learning_rate": lr}
                for d in (4, 6, 8) for lr in (0.05, 0.1)],
    "LightGBM": [{"num_leaves": nl, "learning_rate": lr}
                 for nl in (15, 31, 63) for lr in (0.05, 0.1)],
}

# ---------- Tune on training years only (spatial-block CV) ----------
best, tuning = {}, []
for name, grid in GRIDS.items():
    scores = []
    for params in grid:
        aucs = []
        for fit, val in cv:
            m = make(name, params, y_tr[fit])
            m.fit(X_tr[fit], y_tr[fit])
            aucs.append(roc_auc_score(y_tr[val], m.predict_proba(X_tr[val])[:, 1]))
        scores.append((np.mean(aucs), params))
        tuning.append({"model": name, **params, "cv_auc": round(np.mean(aucs), 4),
                       "cv_sd": round(np.std(aucs), 4)})
        print(name, params, round(np.mean(aucs), 4), flush=True)
    best[name] = max(scores, key=lambda s: s[0])[1]
    print("BEST", name, best[name], flush=True)

pd.DataFrame(tuning).to_csv(f"{OUT}/tuning.csv", index=False)

# ---------- Fit once on all training rows, score test years once ----------
models = {}
scores_te = {}
rf = joblib.load(RF_MODEL)
models["RandomForest"] = rf
scores_te["RandomForest"] = rf.predict_proba(X_te)[:, 1]
for name in GRIDS:
    m = make(name, best[name], y_tr)
    m.fit(X_tr, y_tr)
    models[name] = m
    scores_te[name] = m.predict_proba(X_te)[:, 1]

# ---------- MCD64A1 metrics ----------
blocks = test.block_id.to_numpy()
block_rows = {b: np.where(blocks == b)[0] for b in np.unique(blocks)}
block_ids = list(block_rows)


def top20(y, s):
    """Share of burned cells inside the highest-scored 20% of cells."""
    n = int(round(len(s) * 0.2))
    top = np.argsort(s)[::-1][:n]
    return 100 * y[top].sum() / y.sum()


def block_boot(y, s, seed=SEED):
    """95% interval for AUC, resampling the test blocks with replacement."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(block_ids), size=len(block_ids), replace=True)
        idx = np.concatenate([block_rows[block_ids[i]] for i in pick])
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def calibration(y, s):
    """Brier score, its skill against a constant at the test burned share, and calibration slope."""
    brier = np.mean((s - y) ** 2)
    base = y.mean()
    brier_const = base * (1 - base)
    logit = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    slope = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), y).coef_[0][0]
    return brier, 1 - brier / brier_const, slope


# ---------- FIRMS, per season ----------
det = pd.read_csv(FIRMS, parse_dates=["acq_datetime"])
det = det[(det.conf >= 75) & det.acq_datetime.dt.year.between(2016, 2024)].copy()
det["year"] = det.acq_datetime.dt.year
to_utm = Transformer.from_crs(4326, 32736, always_xy=True)
det["x"], det["y"] = to_utm.transform(det.LONGITUDE.to_numpy(), det.LATITUDE.to_numpy())

firms_hit = np.zeros(len(test), dtype=int)
firms_near = np.zeros(len(test), dtype=bool)  # within 500 m of a detection that season
for year, rows in test.groupby("year").groups.items():
    rows = np.array(list(rows))
    cx = test.loc[rows, "x"].to_numpy()
    cy = test.loc[rows, "y"].to_numpy()
    d = det[det.year == year]
    if len(d) == 0:
        continue
    keys_det = set(zip(np.floor(d.x / CELL).astype(int), np.floor(d.y / CELL).astype(int)))
    keys_cell = list(zip(np.floor(cx / CELL).astype(int), np.floor(cy / CELL).astype(int)))
    firms_hit[rows] = [k in keys_det for k in keys_cell]
    tree = cKDTree(np.c_[d.x.to_numpy(), d.y.to_numpy()])
    dist, _ = tree.query(np.c_[cx, cy])
    firms_near[rows] = dist <= 500.0

keep = ~(firms_near & (firms_hit == 0))  # buffered: drop non-fire cells near a detection
print("FIRMS cell-seasons hit:", firms_hit.sum(), "of", len(test))
print("Buffered sample rows:", keep.sum(), "hits kept:", firms_hit[keep].sum())


def firms_boot(s, mask, seed=SEED):
    """Block bootstrap of the FIRMS AUC on the masked rows."""
    rng = np.random.default_rng(seed)
    vals = []
    yy, ss, bb = firms_hit[mask], s[mask], blocks[mask]
    rows_by_block = {b: np.where(bb == b)[0] for b in np.unique(bb)}
    ids = list(rows_by_block)
    for _ in range(N_BOOT):
        pick = rng.choice(len(ids), size=len(ids), replace=True)
        idx = np.concatenate([rows_by_block[ids[i]] for i in pick])
        if len(np.unique(yy[idx])) == 2:
            vals.append(roc_auc_score(yy[idx], ss[idx]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# ---------- Results ----------
results = []
for name, s in scores_te.items():
    auc = roc_auc_score(y_te, s)
    lo, hi = block_boot(y_te, s)
    brier, skill, slope = calibration(y_te, s)
    f_all = roc_auc_score(firms_hit, s)
    f_lo, f_hi = firms_boot(s, np.ones(len(s), bool))
    f_buf = roc_auc_score(firms_hit[keep], s[keep])
    b_lo, b_hi = firms_boot(s, keep)
    results.append({
        "model": name,
        "params": str(best.get(name, "as trained, 2002 to 2015")),
        "mcd64a1_auc": round(auc, 4),
        "mcd64a1_ci": f"{lo:.3f} to {hi:.3f}",
        "top20_capture_pct": round(top20(y_te, s), 1),
        "brier": round(brier, 4),
        "brier_skill": round(skill, 3),
        "calib_slope": round(slope, 3),
        "firms_auc_no_buffer": round(f_all, 4),
        "firms_ci_no_buffer": f"{f_lo:.3f} to {f_hi:.3f}",
        "firms_auc_buffer": round(f_buf, 4),
        "firms_ci_buffer": f"{b_lo:.3f} to {b_hi:.3f}",
    })
    print(results[-1], flush=True)

pd.DataFrame(results).to_csv(f"{OUT}/three_model_results.csv", index=False)
print("done")

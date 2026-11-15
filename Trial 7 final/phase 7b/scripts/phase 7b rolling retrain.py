"""Phase 7b: rolling retraining, a post-test extension.

Question: does the model fade over 2016 to 2024 because it goes out of date?

For each test year t, a new Random Forest is trained on every row from years
before t, which includes the earlier test years, and scores year t. This is how
a real early-warning system would run, because last year's fires are known.

The settings are the 6a settings. Nothing is tuned again. The headline result
stays the static model scored in Phase 7a. This is a diagnostic next to it, and
it was decided after the test score was known.

Writes to outputs/:
  rolling_results.csv      static and rolling AUC by year, plus pooled values
  rolling_predictions.csv  score for every test cell and season
  rolling_vs_static.png    figure comparing the two
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ---------- Settings ----------
SEED = 42
TREES = 300
FIRST_TEST_YEAR = 2016
N_JOBS = max(1, (os.cpu_count() or 2) // 2)

# Paths, relative to the Phase 7b folder
BASE = Path(__file__).resolve().parents[1]
TABLE = BASE / "data" / "modelling_table.csv"
BEST = BASE / "data" / "best_settings.csv"
STATIC = BASE / "data" / "test_results.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

RESULTS_FILE = OUT / "rolling_results.csv"

if RESULTS_FILE.exists():
    raise RuntimeError("rolling_results.csv already exists. Delete it to run again.")

FEATURES = [
    "dem", "slope", "aspect_east", "aspect_north",
    "rivers_dist", "roads_dist", "settlement_dist",
    "rain_anomaly", "ndvi_peak", "ndmi_anomaly", "lst_anomaly", "lulc",
    "burn_count_3yr", "years_since_burn",
]


def read_setting(value):
    """Settings come back from the CSV as text or numbers."""
    try:
        number = float(value)
        return int(number) if number.is_integer() and number > 1 else number
    except (TypeError, ValueError):
        return value


# ---------- Settings of the 6a winner ----------
best = pd.read_csv(BEST, dtype=str).iloc[0]
params = {
    "max_depth": read_setting(best["max_depth"]),
    "min_samples_leaf": read_setting(best["min_samples_leaf"]),
    "max_features": read_setting(best["max_features"]),
    "max_samples": read_setting(best["max_samples"]),
}
print("6a settings:", params)

# ---------- Data ----------
table = pd.read_csv(TABLE)
years = table["year"].to_numpy()
X = table[FEATURES].to_numpy()
y = table["burned"].to_numpy()
static = pd.read_csv(STATIC, dtype={"year": str})


def top_capture(truth, scores, share=0.20):
    """Share of burned cells that fall in the highest-scored 20% of cells."""
    n_top = int(round(len(scores) * share))
    top = np.argsort(scores)[::-1][:n_top]
    return 100 * truth[top].sum() / truth.sum()


# ---------- Rolling retraining ----------
rows = []
predictions = []

for year in range(FIRST_TEST_YEAR, int(years.max()) + 1):
    fit_idx = np.where(years < year)[0]
    val_idx = np.where(years == year)[0]

    model = RandomForestClassifier(
        n_estimators=TREES,
        class_weight="balanced_subsample",
        n_jobs=N_JOBS,
        random_state=SEED,
        **params,
    ).fit(X[fit_idx], y[fit_idx])

    score = model.predict_proba(X[val_idx])[:, 1]

    rows.append({
        "year": str(year),
        "train_rows": len(fit_idx),
        "static_auc": float(static.loc[static["year"] == str(year), "auc"].iloc[0]),
        "rolling_auc": round(roc_auc_score(y[val_idx], score), 4),
        "rolling_top20_capture_pct": round(top_capture(y[val_idx], score), 1),
    })

    predictions.append(pd.DataFrame({
        "cell_id": table["cell_id"].to_numpy()[val_idx],
        "year": year,
        "burned": y[val_idx],
        "score": score.round(5),
    }))

    print(f"{year}: rolling {rows[-1]['rolling_auc']}  static {rows[-1]['static_auc']}")

predictions = pd.concat(predictions, ignore_index=True)
predictions.to_csv(OUT / "rolling_predictions.csv", index=False)

# ---------- Summary ----------
results = pd.DataFrame(rows)
results["difference"] = (results["rolling_auc"] - results["static_auc"]).round(4)

pooled_rolling = roc_auc_score(predictions["burned"], predictions["score"])
pooled_static = float(static.loc[static["year"].str.startswith("ALL"), "auc"].iloc[0])

results = pd.concat([results, pd.DataFrame([{
    "year": "POOLED",
    "train_rows": np.nan,
    "static_auc": pooled_static,
    "rolling_auc": round(pooled_rolling, 4),
    "rolling_top20_capture_pct": round(results["rolling_top20_capture_pct"].mean(), 1),
    "difference": round(pooled_rolling - pooled_static, 4),
}])], ignore_index=True)

results.to_csv(RESULTS_FILE, index=False)

yearly = results.iloc[:-1]
print(results.to_string(index=False))
print(f"\nMean yearly AUC: static {yearly['static_auc'].mean():.4f}, rolling {yearly['rolling_auc'].mean():.4f}")

# ---------- Figure ----------
x = np.arange(len(yearly))
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.bar(x - 0.2, yearly["static_auc"], width=0.4, color="#dd6b20", label="Static model (trained on 2002 to 2015)")
ax.bar(x + 0.2, yearly["rolling_auc"], width=0.4, color="#2b6cb0", label="Rolling model (retrained each year)")
ax.axhline(0.5, color="grey", linestyle="--", label="Chance (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(yearly["year"])
ax.set_ylim(0.4, 0.9)
ax.set_xlabel("Test year")
ax.set_ylabel("AUC")
ax.set_title("Phase 7b: static model against yearly retraining (post-test extension)")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "rolling_vs_static.png", dpi=300, bbox_inches="tight")
plt.close(fig)
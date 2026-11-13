import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier      # needed to load the saved XGBoost model
from lightgbm import LGBMClassifier    # needed to load the saved LightGBM model

# Folder layout (relative to this script):
#   Phase_6_validation/
#       phase6_validation.py
#       data/table/training_table.csv            main table, 500 m (Phase 4 output)
#       data/sensitivity/training_table_0m.csv   Phase 4 run at 0 m
#       data/sensitivity/training_table_250m.csv Phase 4 run at 250 m
#       data/sensitivity/training_table_1000m.csv Phase 4 run at 1000 m
#       data/models/                             the 3 saved models (Phase 5 outputs/models)
#       outputs/                                 test_performance.csv, importance.csv,
#                                                buffer_sensitivity.csv

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_TABLE = os.path.join(HERE, 'data', 'table', 'training_table.csv')
SENS_DIR = os.path.join(HERE, 'data', 'sensitivity')
MODEL_DIR = os.path.join(HERE, 'data', 'models')
OUT_DIR = os.path.join(HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
THRESHOLD = 0.5
N_BOOT = 1000
N_PERM = 10
MAIN_MODEL = 'RandomForest'
COMPARISON_MODELS = ['XGBoost', 'LightGBM']
SENSITIVITY_BUFFERS = {0: 'training_table_0m.csv', 250: 'training_table_250m.csv',
                       1000: 'training_table_1000m.csv'}

PREDICTORS = [
    'ndvi', 'ndmi', 'lst', 'rainfall',
    'rivers_dist', 'roads_dist', 'settlement_dist',
    'aspect_east', 'aspect_north', 'dem', 'lulc', 'slope',
]

# Same settings as Phase 5. Changing them here would make the buffer runs
# differ from the main model.
RF_SETTINGS = dict(n_estimators=300, max_depth=20, class_weight='balanced',
                   random_state=SEED, n_jobs=-1)


def scores(y_true, prob):
    pred = (prob >= THRESHOLD).astype(int)
    return {
        'auc': roc_auc_score(y_true, prob),
        'f1': f1_score(y_true, pred),
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
    }


def bootstrap_auc_ci(y_true, prob, rng):
    """95 percent interval for the test AUC, by resampling test rows."""
    n = len(y_true)
    values = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) == 2:
            values.append(roc_auc_score(y_true[idx], prob[idx]))
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def split(table):
    """Return the training rows and the test rows, as X and y."""
    train = table[table['period'] == 'train']
    test = table[table['period'] == 'test']
    assert set(test['period']) == {'test'} and set(train['period']) == {'train'}
    return (train[PREDICTORS].to_numpy(), train['label'].to_numpy(),
            test[PREDICTORS].to_numpy(), test['label'].to_numpy())


rng = np.random.default_rng(SEED)
table = pd.read_csv(MAIN_TABLE)
X_train, y_train, X_test, y_test = split(table)
print(f'Main table: {len(X_train)} training rows, {len(X_test)} test rows')

# ---------- 1. score each saved model once on the test rows ----------
perf_rows = []
loaded = {}
for name in [MAIN_MODEL] + COMPARISON_MODELS:
    model = joblib.load(os.path.join(MODEL_DIR, f'{name}.joblib'))
    loaded[name] = model
    prob = model.predict_proba(X_test)[:, 1]
    lo, hi = bootstrap_auc_ci(y_test, prob, rng)
    perf_rows.append({
        'model': name,
        'role': 'main' if name == MAIN_MODEL else 'comparison',
        'n_test': len(y_test),
        **scores(y_test, prob),
        'auc_ci_low': round(lo, 4),
        'auc_ci_high': round(hi, 4),
    })

performance = pd.DataFrame(perf_rows)
performance.to_csv(os.path.join(OUT_DIR, 'test_performance.csv'), index=False)

# ---------- 2. permutation importance for the main model, on the test rows ----------
rf = loaded[MAIN_MODEL]
perm = permutation_importance(rf, X_test, y_test, scoring='roc_auc',
                              n_repeats=N_PERM, random_state=SEED, n_jobs=-1)
importance = pd.DataFrame({
    'feature': PREDICTORS,
    'importance_mean': perm.importances_mean,
    'importance_std': perm.importances_std,
}).sort_values('importance_mean', ascending=False).reset_index(drop=True)
importance['rank'] = range(1, len(importance) + 1)
importance.to_csv(os.path.join(OUT_DIR, 'importance.csv'), index=False)

# ---------- 3. buffer sensitivity: retrain and score at each buffer ----------
sens_rows = []
main_row = performance.loc[performance['model'] == MAIN_MODEL].iloc[0]
sens_rows.append({
    'buffer_m': 500, 'source': 'main table', 'n_train': len(y_train), 'n_test': len(y_test),
    'auc': round(main_row['auc'], 4), 'f1': round(main_row['f1'], 4),
    'accuracy': round(main_row['accuracy'], 4), 'precision': round(main_row['precision'], 4),
    'recall': round(main_row['recall'], 4),
    'auc_ci_low': main_row['auc_ci_low'], 'auc_ci_high': main_row['auc_ci_high'],
})

missing = []
for buffer_m, fname in SENSITIVITY_BUFFERS.items():
    path = os.path.join(SENS_DIR, fname)
    if not os.path.exists(path):
        missing.append(f'{buffer_m} m ({fname})')
        continue
    sens_table = pd.read_csv(path)
    Xtr, ytr, Xte, yte = split(sens_table)
    model = RandomForestClassifier(**RF_SETTINGS).fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    lo, hi = bootstrap_auc_ci(yte, prob, rng)
    sens_rows.append({
        'buffer_m': buffer_m, 'source': fname, 'n_train': len(ytr), 'n_test': len(yte),
        **{k: round(v, 4) for k, v in scores(yte, prob).items()},
        'auc_ci_low': round(lo, 4), 'auc_ci_high': round(hi, 4),
    })

sensitivity = pd.DataFrame(sens_rows).sort_values('buffer_m')
sensitivity.to_csv(os.path.join(OUT_DIR, 'buffer_sensitivity.csv'), index=False)

print('\nPhase 6: test performance (2016 to 2024, not used in training)')
print('=' * 70)
print(performance.round(4).to_string(index=False))
print('\nPermutation importance (main model, test rows)')
print(importance.round(4).to_string(index=False))
print('\nBuffer sensitivity (Random Forest retrained at each buffer)')
print(sensitivity.round(4).to_string(index=False))
if missing:
    print('\nNOT RUN, missing tables:', '; '.join(missing))
    print('Run Phase 4 at each missing buffer and copy its training_table.csv into data/sensitivity/.')
print('-' * 70)
print('RESULT: complete. Outputs written to outputs/.')
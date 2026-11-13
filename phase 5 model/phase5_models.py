import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Folder layout (relative to this script):
#   Phase_5_models/
#       phase5_models.py
#       data/table/training_table.csv    (Phase 4 output)
#       outputs/                         cross_validation.csv, model_performance.csv,
#                                        model_settings.csv
#       outputs/models/                  the three fitted models

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_FILE = os.path.join(HERE, 'data', 'table', 'training_table.csv')
OUT_DIR = os.path.join(HERE, 'outputs')
MODEL_DIR = os.path.join(OUT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
N_FOLDS = 5
THRESHOLD = 0.5

PREDICTORS = [
    'ndvi', 'ndmi', 'lst', 'rainfall',
    'rivers_dist', 'roads_dist', 'settlement_dist',
    'aspect_east', 'aspect_north', 'dem', 'lulc', 'slope',
]

# Settings as they were run in the dissertation analysis. No grid search was run,
# so these are the fixed values and the dissertation must say so.
MODEL_SETTINGS = {
    'RandomForest': dict(n_estimators=300, max_depth=20, class_weight='balanced',
                         random_state=SEED, n_jobs=-1),
    'XGBoost': dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                    random_state=SEED, n_jobs=-1),
    'LightGBM': dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                     random_state=SEED, n_jobs=-1, verbose=-1),
}


def build(name):
    if name == 'RandomForest':
        return RandomForestClassifier(**MODEL_SETTINGS[name])
    if name == 'XGBoost':
        return XGBClassifier(**MODEL_SETTINGS[name])
    return LGBMClassifier(**MODEL_SETTINGS[name])


table = pd.read_csv(TABLE_FILE)
train = table[table['period'] == 'train'].reset_index(drop=True)
assert set(train['period']) == {'train'}, 'test rows must not be used in Phase 5'

X = train[PREDICTORS].to_numpy()
y = train['label'].to_numpy()
groups = train['block_id'].to_numpy()


def scores(y_true, prob):
    pred = (prob >= THRESHOLD).astype(int)
    return {
        'auc': roc_auc_score(y_true, prob),
        'f1': f1_score(y_true, pred),
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
    }


schemes = {
    'spatial_block': list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups)),
    'random': list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                   random_state=SEED).split(X, y)),
}

fold_rows = []
for name in MODEL_SETTINGS:
    for scheme, splits in schemes.items():
        for fold, (tr, va) in enumerate(splits, start=1):
            model = build(name)
            model.fit(X[tr], y[tr])
            prob = model.predict_proba(X[va])[:, 1]
            fold_rows.append({'model': name, 'scheme': scheme, 'fold': fold,
                              **scores(y[va], prob)})
            print(f'{name:12s} {scheme:14s} fold {fold}  AUC {fold_rows[-1]["auc"]:.4f}')

cv = pd.DataFrame(fold_rows)
cv.to_csv(os.path.join(OUT_DIR, 'cross_validation.csv'), index=False)

metric_cols = ['auc', 'f1', 'accuracy', 'precision', 'recall']
summary = (cv.groupby(['model', 'scheme'])[metric_cols]
             .agg(['mean', 'std']))
summary.columns = [f'{m}_{s}' for m, s in summary.columns]
summary = summary.reset_index()
summary.to_csv(os.path.join(OUT_DIR, 'model_performance.csv'), index=False)

# Final models: fitted on all training rows, saved for Phase 6 and Phase 7.
for name in MODEL_SETTINGS:
    model = build(name)
    model.fit(X, y)
    joblib.dump(model, os.path.join(MODEL_DIR, f'{name}.joblib'))

pd.DataFrame([
    {'model': name, 'setting': k, 'value': v}
    for name, cfg in MODEL_SETTINGS.items() for k, v in cfg.items()
]).to_csv(os.path.join(OUT_DIR, 'model_settings.csv'), index=False)

print('\nPhase 5: model performance on the training years (cross-validation)')
print('=' * 70)
print(summary.round(4).to_string(index=False))
print('-' * 70)
print(f'Training rows: {len(train)} ({int(y.sum())} fire, {int((y == 0).sum())} non-fire)')
print('RESULT: complete. Test rows were not used. Models saved to outputs/models/.')
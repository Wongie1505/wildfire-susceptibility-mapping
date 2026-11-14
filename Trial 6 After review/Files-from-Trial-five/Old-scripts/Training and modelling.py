# # ================================================
# # VIPHYA WILDFIRE SUSCEPTIBILITY - MODELING
# # Input: viphya_sampled_data.csv
# # Output: AUC scores, ROC curve, feature importance
# # ================================================

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.metrics import (roc_auc_score, roc_curve,
#                              classification_report, confusion_matrix,
#                              ConfusionMatrixDisplay)
# from sklearn.preprocessing import LabelEncoder
# import xgboost as xgb
# import joblib

# # ================================================
# # 1. LOAD DATA
# # ================================================

# csv_path = r"C:\Users\wongi\OneDrive\Desktop\Year 3\Sem 2\My Research\Project and thesis\Trial_five\DATA\viphya_sampled_data.csv"

# df = pd.read_csv(csv_path)

# print("Shape:", df.shape)
# print("\nLabel distribution:")
# print(df['fire_label'].value_counts())
# print("\nColumns:", df.columns.tolist())

# # ================================================
# # 2. PREPARE FEATURES AND LABEL
# # ================================================

# # Encode LULC if it's categorical
# if df['LULC'].dtype == object:
#     le = LabelEncoder()
#     df['LULC'] = le.fit_transform(df['LULC'].astype(str))

# target   = 'fire_label'
# features = [c for c in df.columns if c != target]

# X = df[features]
# y = df[target]

# print("\nFeatures:", features)

# # ================================================
# # 3. TRAIN / TEST SPLIT  (70% train, 30% test)
# # stratified = keeps same fire/nonfire ratio in both
# # ================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.30,
#     random_state=42,
#     stratify=y
# )

# print(f"\nTrain rows : {len(X_train)}")
# print(f"Test rows  : {len(X_test)}")
# print(f"Train label balance: {y_train.value_counts().to_dict()}")
# print(f"Test  label balance: {y_test.value_counts().to_dict()}")

# # ================================================
# # 4. RANDOM FOREST
# # ================================================

# print("\n--- Training Random Forest ---")

# rf = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=20,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     max_features='sqrt',
#     class_weight='balanced',
#     n_jobs=-1,
#     random_state=42
# )

# rf.fit(X_train, y_train)

# rf_proba = rf.predict_proba(X_test)[:, 1]
# rf_pred  = rf.predict(X_test)
# rf_auc   = roc_auc_score(y_test, rf_proba)

# print(f"Random Forest AUC : {rf_auc:.4f}")
# print("\nRF Classification Report:")
# print(classification_report(y_test, rf_pred,
#       target_names=['No Fire', 'Fire']))

# # ================================================
# # 5. XGBOOST
# # ================================================

# print("\n--- Training XGBoost ---")

# scale_w = (y_train == 0).sum() / (y_train == 1).sum()

# xgb_model = xgb.XGBClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     scale_pos_weight=scale_w,
#     eval_metric='logloss',
#     random_state=42,
#     n_jobs=-1
# )

# xgb_model.fit(
#     X_train, y_train,
#     eval_set=[(X_test, y_test)],
#     verbose=False
# )

# xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
# xgb_pred  = xgb_model.predict(X_test)
# xgb_auc   = roc_auc_score(y_test, xgb_proba)

# print(f"XGBoost AUC : {xgb_auc:.4f}")
# print("\nXGB Classification Report:")
# print(classification_report(y_test, xgb_pred,
#       target_names=['No Fire', 'Fire']))

# # ================================================
# # 6. 5-FOLD CROSS VALIDATION
# # ================================================

# print("\n--- 5-Fold Stratified Cross Validation ---")

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# rf_cv  = cross_val_score(rf,        X, y, cv=cv,
#                           scoring='roc_auc', n_jobs=-1)
# xgb_cv = cross_val_score(xgb_model, X, y, cv=cv,
#                           scoring='roc_auc', n_jobs=-1)

# print(f"RF  CV AUC : {rf_cv.mean():.4f}  (+/- {rf_cv.std():.4f})")
# print(f"XGB CV AUC : {xgb_cv.mean():.4f}  (+/- {xgb_cv.std():.4f})")

# # ================================================
# # 7. ROC CURVE PLOT
# # ================================================

# fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_proba)
# fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr_rf,  tpr_rf,
#          label=f'Random Forest (AUC = {rf_auc:.3f})', lw=2)
# plt.plot(fpr_xgb, tpr_xgb,
#          label=f'XGBoost      (AUC = {xgb_auc:.3f})', lw=2)
# plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', lw=1)
# plt.xlabel('False Positive Rate', fontsize=12)
# plt.ylabel('True Positive Rate', fontsize=12)
# plt.title('ROC Curves — Viphya Wildfire Susceptibility', fontsize=13)
# plt.legend(fontsize=11)
# plt.tight_layout()
# plt.savefig('roc_curves.png', dpi=300)
# plt.show()
# print("\nROC curve saved as roc_curves.png")

# # ================================================
# # 8. FEATURE IMPORTANCE
# # ================================================

# rf_imp = pd.DataFrame({
#     'Feature'   : features,
#     'Importance': rf.feature_importances_
# }).sort_values('Importance', ascending=False)

# xgb_imp = pd.DataFrame({
#     'Feature'   : features,
#     'Importance': xgb_model.feature_importances_
# }).sort_values('Importance', ascending=False)

# print("\nRF Feature Importance:")
# print(rf_imp.to_string(index=False))

# print("\nXGB Feature Importance:")
# print(xgb_imp.to_string(index=False))

# # Plot RF importance
# plt.figure(figsize=(9, 6))
# plt.barh(rf_imp['Feature'], rf_imp['Importance'], color='steelblue')
# plt.xlabel('Importance', fontsize=12)
# plt.title('Random Forest — Feature Importance', fontsize=13)
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=300)
# plt.show()

# # ================================================
# # 9. CONFUSION MATRIX
# # ================================================

# best_name  = 'XGBoost' if xgb_auc >= rf_auc else 'Random Forest'
# best_pred  = xgb_pred  if xgb_auc >= rf_auc else rf_pred
# best_auc   = max(xgb_auc, rf_auc)

# cm = confusion_matrix(y_test, best_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                                display_labels=['No Fire', 'Fire'])
# disp.plot(cmap='Blues')
# plt.title(f'Confusion Matrix — {best_name}', fontsize=13)
# plt.tight_layout()
# plt.savefig('confusion_matrix.png', dpi=300)
# plt.show()

# # ================================================
# # 10. SAVE BEST MODEL
# # ================================================

# best_model = xgb_model if xgb_auc >= rf_auc else rf

# joblib.dump(best_model, 'best_wildfire_model.pkl')
# joblib.dump(features,   'model_features.pkl')

# print(f"\n{'='*50}")
# print(f"BEST MODEL  : {best_name}")
# print(f"AUC SCORE   : {best_auc:.4f}")
# print(f"CV AUC      : {(xgb_cv if best_name == 'XGBoost' else rf_cv).mean():.4f}")
# print(f"{'='*50}")
# print("\nModel saved. Next step: predict susceptibility map.")





# ================================================
# VIPHYA WILDFIRE SUSCEPTIBILITY - MODELING
# Input: viphya_sampled_data.csv
# Output: AUC scores, ROC curve, feature importance
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import joblib

# ================================================
# 1. LOAD DATA
# ================================================

csv_path = r"C:\Users\wongi\OneDrive\Desktop\Year 3\Sem 2\My Research\Project and thesis\Trial_five\DATA\viphya_sampled_data.csv"

df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("\nLabel distribution:")
print(df['fire_label'].value_counts())
print("\nColumns:", df.columns.tolist())

# ================================================
# 2. PREPARE FEATURES AND LABEL
# ================================================

if df['LULC'].dtype == object:
    le = LabelEncoder()
    df['LULC'] = le.fit_transform(df['LULC'].astype(str))

target   = 'fire_label'
features = [c for c in df.columns if c != target]

X = df[features]
y = df[target]

print("\nFeatures:", features)

# ================================================
# 3. TRAIN / TEST SPLIT
# ================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print(f"\nTrain rows : {len(X_train)}")
print(f"Test rows  : {len(X_test)}")
print(f"Train label balance: {y_train.value_counts().to_dict()}")
print(f"Test  label balance: {y_test.value_counts().to_dict()}")

# ================================================
# 4. RANDOM FOREST
# ================================================

print("\n--- Training Random Forest ---")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred  = rf.predict(X_test)
rf_auc   = roc_auc_score(y_test, rf_proba)

print(f"Random Forest AUC : {rf_auc:.4f}")
print("\nRF Classification Report:")
print(classification_report(y_test, rf_pred,
      target_names=['No Fire', 'Fire']))

# ================================================
# 5. XGBOOST
# ================================================

print("\n--- Training XGBoost ---")

scale_w = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_w,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred  = xgb_model.predict(X_test)
xgb_auc   = roc_auc_score(y_test, xgb_proba)

print(f"XGBoost AUC : {xgb_auc:.4f}")
print("\nXGB Classification Report:")
print(classification_report(y_test, xgb_pred,
      target_names=['No Fire', 'Fire']))

# ================================================
# 6. LIGHTGBM
# ================================================

print("\n--- Training LightGBM ---")

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_w,
    n_jobs=-1,
    random_state=42,
    verbose=-1          # suppress training output
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)]
)

lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_pred  = lgb_model.predict(X_test)
lgb_auc   = roc_auc_score(y_test, lgb_proba)

print(f"LightGBM AUC : {lgb_auc:.4f}")
print("\nLGB Classification Report:")
print(classification_report(y_test, lgb_pred,
      target_names=['No Fire', 'Fire']))

# ================================================
# 7. 5-FOLD CROSS VALIDATION
# ================================================

print("\n--- 5-Fold Stratified Cross Validation ---")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_cv  = cross_val_score(rf,        X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
xgb_cv = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
lgb_cv = cross_val_score(lgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"RF  CV AUC : {rf_cv.mean():.4f}  (+/- {rf_cv.std():.4f})")
print(f"XGB CV AUC : {xgb_cv.mean():.4f}  (+/- {xgb_cv.std():.4f})")
print(f"LGB CV AUC : {lgb_cv.mean():.4f}  (+/- {lgb_cv.std():.4f})")

# ================================================
# 8. ROC CURVE PLOT — ALL THREE MODELS
# ================================================

fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf,  tpr_rf,
         label=f'Random Forest (AUC = {rf_auc:.3f})',  lw=2)
plt.plot(fpr_xgb, tpr_xgb,
         label=f'XGBoost      (AUC = {xgb_auc:.3f})', lw=2)
plt.plot(fpr_lgb, tpr_lgb,
         label=f'LightGBM     (AUC = {lgb_auc:.3f})', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', lw=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves — Viphya Wildfire Susceptibility', fontsize=13)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300)
plt.show()
print("\nROC curve saved.")

# ================================================
# 9. FEATURE IMPORTANCE — ALL THREE
# ================================================

rf_imp = pd.DataFrame({
    'Feature'   : features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

xgb_imp = pd.DataFrame({
    'Feature'   : features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

lgb_imp = pd.DataFrame({
    'Feature'   : features,
    'Importance': lgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRF Feature Importance:")
print(rf_imp.to_string(index=False))

print("\nXGB Feature Importance:")
print(xgb_imp.to_string(index=False))

print("\nLGB Feature Importance:")
print(lgb_imp.to_string(index=False))

# Plot RF importance (RF is your best model so far)
plt.figure(figsize=(9, 6))
plt.barh(rf_imp['Feature'], rf_imp['Importance'], color='steelblue')
plt.xlabel('Importance', fontsize=12)
plt.title('Random Forest — Feature Importance', fontsize=13)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=300)
plt.show()

# ================================================
# 10. CONFUSION MATRIX — BEST MODEL
# ================================================

all_aucs  = {'Random Forest': rf_auc,
             'XGBoost'      : xgb_auc,
             'LightGBM'     : lgb_auc}

best_name = max(all_aucs, key=all_aucs.get)
best_auc  = all_aucs[best_name]
best_pred = {'Random Forest': rf_pred,
             'XGBoost'      : xgb_pred,
             'LightGBM'     : lgb_pred}[best_name]

cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['No Fire', 'Fire'])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix — {best_name}', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# ================================================
# 11. SAVE ALL THREE MODELS SEPARATELY
# ================================================

joblib.dump(rf,        'model_rf.pkl')
joblib.dump(xgb_model, 'model_xgb.pkl')
joblib.dump(lgb_model, 'model_lgb.pkl')
joblib.dump(features,  'model_features.pkl')

print("\nAll three models saved:")
print("  model_rf.pkl")
print("  model_xgb.pkl")
print("  model_lgb.pkl")
print("  model_features.pkl")

# ================================================
# 12. SUMMARY TABLE
# ================================================

summary = pd.DataFrame({
    'Model'     : ['Random Forest', 'XGBoost', 'LightGBM'],
    'AUC'       : [round(rf_auc,  4),
                   round(xgb_auc, 4),
                   round(lgb_auc, 4)],
    'CV AUC'    : [round(rf_cv.mean(),  4),
                   round(xgb_cv.mean(), 4),
                   round(lgb_cv.mean(), 4)],
    'CV Std'    : [round(rf_cv.std(),  4),
                   round(xgb_cv.std(), 4),
                   round(lgb_cv.std(), 4)]
})

print(f"\n{'='*55}")
print("MODEL COMPARISON SUMMARY")
print('='*55)
print(summary.to_string(index=False))
print('='*55)
print(f"BEST MODEL: {best_name}  |  AUC: {best_auc:.4f}")
print('='*55)
print("\nNext step: use saved models to predict susceptibility maps.")


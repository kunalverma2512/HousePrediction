import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("Starting the Massive Triple Blend Ensemble Training Process...")
print("============================================================")

print("[SYSTEM] Loading preprocessed data from disk...")
train = pd.read_csv('../data/processed/cleaned_train.csv')
test = pd.read_csv('../data/processed/cleaned_test.csv')
test_ids = pd.read_csv('../data/processed/test_ids.csv')['Id']

X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Define K-Fold for verbose logging
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize Models
xgb = XGBRegressor(learning_rate=0.05, n_estimators=1000, max_depth=3, random_state=42, verbosity=0)
lgbm = LGBMRegressor(learning_rate=0.05, n_estimators=1000, max_depth=3, random_state=42, verbose=-1)
lasso = Lasso(alpha=0.0005, random_state=42)

print("\n[PHASE 1] Training XGBoost Regressor...")
xgb_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    xgb.fit(X_tr, y_tr)
    preds = xgb.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val, preds))
    xgb_scores.append(score)
    print(f"   -> Fold {fold+1} completed. Root Mean Squared Error: {score:.4f}")
print(f"[SUCCESS] XGBoost Average Error: {np.mean(xgb_scores):.4f}")

print("\n[PHASE 2] Training LightGBM Regressor...")
lgbm_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgbm.fit(X_tr, y_tr)
    preds = lgbm.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val, preds))
    lgbm_scores.append(score)
    print(f"   -> Fold {fold+1} completed. Root Mean Squared Error: {score:.4f}")
print(f"[SUCCESS] LightGBM Average Error: {np.mean(lgbm_scores):.4f}")

print("\n[PHASE 3] Training Lasso Regression (Linear Model)...")
lasso_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lasso.fit(X_tr, y_tr)
    preds = lasso.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val, preds))
    lasso_scores.append(score)
    print(f"   -> Fold {fold+1} completed. Root Mean Squared Error: {score:.4f}")
print(f"[SUCCESS] Lasso Average Error: {np.mean(lasso_scores):.4f}")

print("\n[PHASE 4] Retraining all models on the entire dataset for final predictions...")
xgb.fit(X, y)
print("   -> XGBoost full train complete.")
lgbm.fit(X, y)
print("   -> LightGBM full train complete.")
lasso.fit(X, y)
print("   -> Lasso full train complete.")

print("\n[PHASE 5] Predicting on Test Data and Blending...")
xgb_final_preds = np.expm1(xgb.predict(test))
lgbm_final_preds = np.expm1(lgbm.predict(test))
lasso_final_preds = np.expm1(lasso.predict(test))

# The Triple Blend!
blended_predictions = (0.34 * xgb_final_preds) + (0.33 * lgbm_final_preds) + (0.33 * lasso_final_preds)
print("   -> Predictions generated and averaged successfully.")

print("\n[PHASE 6] Saving Final Submission...")
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': blended_predictions
})
submission.to_csv('../submissions/submission_triple_blend.csv', index=False)
print("[COMPLETED] File saved as '../submissions/submission_triple_blend.csv'. Ready for Kaggle upload!")

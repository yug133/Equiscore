import pandas as pd
import numpy as np
from data.loader import load_all_tables
from data.preprocessor import preprocess_application
from data.splitter import run_full_split_pipeline
from features.feature_pipeline import compute_all_features, merge_features_with_application

from models.logistic_regression import train_logistic_model, predict_proba_logistic, save_logistic_model
from models.random_forest import train_random_forest, predict_proba_random_forest, save_random_forest
from models.xgboost_standard import train_xgboost, predict_proba_xgboost, save_xgboost
from models.xgboost_fair import train_fair_xgboost, predict_proba_fair_xgboost, save_fair_xgboost
from models.model_evaluator import compute_all_metrics, compare_models

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop remaining object columns and fill NaNs."""
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print(f"Dropping remaining object columns: {obj_cols}")
        df = df.drop(columns=obj_cols)
    df = df.fillna(0)
    return df

print("=" * 60)
print("EQUISCORE MODEL TRAINING PIPELINE")
print("=" * 60)

# ── Step 1: Load data ──────────────────────────────────────────
print("\n[1/6] Loading data...")
tables = load_all_tables()
raw_app = tables["application_train"].copy()

# ── Build sensitive features BEFORE preprocessing ─────────────
print("\n[2/6] Building sensitive features from raw data...")
sensitive_all = (
    raw_app["CODE_GENDER"].astype(str) + "_region" +
    raw_app["REGION_RATING_CLIENT"].astype(str)
).rename("sensitive_group")
sensitive_all.index = raw_app["SK_ID_CURR"]
print(f"Sensitive groups:\n{sensitive_all.value_counts().to_dict()}")

# ── Step 2: Preprocess ─────────────────────────────────────────
print("\n[3/6] Preprocessing...")
df = preprocess_application(raw_app)

# ── Step 3: Compute & merge features ──────────────────────────
print("\n[4/6] Computing custom features...")
feature_df = compute_all_features(tables)
df = merge_features_with_application(df, feature_df)

# ── Final cleanup: remove any remaining non-numeric columns ───
print("\nCleaning non-numeric columns...")
df = clean_dataframe(df)
print(f"Final DataFrame shape: {df.shape}")
print(f"Remaining dtypes: {df.dtypes.value_counts().to_dict()}")

# ── Step 4: Split ──────────────────────────────────────────────
print("\n[5/6] Splitting data...")
X_train, X_test, y_train, y_test = run_full_split_pipeline(df)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# Align sensitive features
sensitive_train = sensitive_all.reindex(X_train.index).fillna("Unknown_region2")
sensitive_test  = sensitive_all.reindex(X_test.index).fillna("Unknown_region2")

results = {}

# ── Logistic Regression ────────────────────────────────────────
print("\n[6/6a] Training Logistic Regression...")
lr_model, lr_scaler = train_logistic_model(X_train, y_train)
lr_proba = predict_proba_logistic(lr_model, X_test, lr_scaler)
save_logistic_model(lr_model, lr_scaler)
results["logistic"] = compute_all_metrics(
    y_test, lr_proba, sensitive_features=sensitive_test,
    model_name="logistic"
)
print(f"✅ Logistic AUC={results['logistic']['auc_roc']:.4f} "
      f"KS={results['logistic']['ks_statistic']:.4f} "
      f"Gini={results['logistic']['gini']:.4f}")

# ── Random Forest ──────────────────────────────────────────────
print("\n[6/6b] Training Random Forest (5-10 mins)...")
rf_model = train_random_forest(X_train, y_train)
rf_proba = predict_proba_random_forest(rf_model, X_test)
save_random_forest(rf_model)
results["random_forest"] = compute_all_metrics(
    y_test, rf_proba, sensitive_features=sensitive_test,
    model_name="random_forest"
)
print(f"✅ Random Forest AUC={results['random_forest']['auc_roc']:.4f} "
      f"KS={results['random_forest']['ks_statistic']:.4f} "
      f"Gini={results['random_forest']['gini']:.4f}")

# ── XGBoost Standard ───────────────────────────────────────────
print("\n[6/6c] Training XGBoost Standard...")
xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
xgb_proba = predict_proba_xgboost(xgb_model, X_test)
save_xgboost(xgb_model)
results["xgboost"] = compute_all_metrics(
    y_test, xgb_proba, sensitive_features=sensitive_test,
    model_name="xgboost"
)
print(f"✅ XGBoost AUC={results['xgboost']['auc_roc']:.4f} "
      f"KS={results['xgboost']['ks_statistic']:.4f} "
      f"Gini={results['xgboost']['gini']:.4f}")

# ── Fair XGBoost ───────────────────────────────────────────────
print("\n[6/6d] Training Fair XGBoost (10-15 mins)...")
fair_model = train_fair_xgboost(X_train, y_train, sensitive_train)
fair_proba = predict_proba_fair_xgboost(fair_model, X_test)
save_fair_xgboost(fair_model)
results["xgboost_fair"] = compute_all_metrics(
    y_test, fair_proba, sensitive_features=sensitive_test,
    model_name="xgboost_fair"
)
print(f"✅ Fair XGBoost AUC={results['xgboost_fair']['auc_roc']:.4f} "
      f"KS={results['xgboost_fair']['ks_statistic']:.4f} "
      f"Gini={results['xgboost_fair']['gini']:.4f}")

# ── Final comparison ───────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
comparison = compare_models(results)
print(comparison.to_string(index=False))
print("\n✅ All models trained and saved successfully!")

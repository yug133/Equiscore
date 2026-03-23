import pandas as pd
import numpy as np
from data.loader import load_all_tables
from data.preprocessor import preprocess_application
from data.splitter import run_full_split_pipeline
from features.feature_pipeline import compute_all_features, merge_features_with_application
from models.xgboost_fair import train_fair_xgboost, predict_proba_fair_xgboost, save_fair_xgboost
from models.model_evaluator import compute_all_metrics

def clean_dataframe(df):
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = df.drop(columns=obj_cols)
    return df.fillna(0)

print("Loading data...")
tables = load_all_tables()
raw_app = tables["application_train"].copy()

sensitive_all = (
    raw_app["CODE_GENDER"].astype(str) + "_region" +
    raw_app["REGION_RATING_CLIENT"].astype(str)
).rename("sensitive_group")
sensitive_all.index = raw_app["SK_ID_CURR"]

print("Preprocessing...")
df = preprocess_application(raw_app)
feature_df = compute_all_features(tables)
df = merge_features_with_application(df, feature_df)
df = clean_dataframe(df)

print("Splitting...")
X_train, X_test, y_train, y_test = run_full_split_pipeline(df)
sensitive_train = sensitive_all.reindex(X_train.index).fillna("Unknown_region2")
sensitive_test  = sensitive_all.reindex(X_test.index).fillna("Unknown_region2")

print("Training Fair XGBoost (10-15 mins)...")
fair_model = train_fair_xgboost(X_train, y_train, sensitive_train)
fair_proba = predict_proba_fair_xgboost(fair_model, X_test)
save_fair_xgboost(fair_model)

metrics = compute_all_metrics(
    y_test, fair_proba,
    sensitive_features=sensitive_test,
    model_name="xgboost_fair"
)
print(f"\n✅ Fair XGBoost AUC={metrics['auc_roc']:.4f} "
      f"KS={metrics['ks_statistic']:.4f} "
      f"Gini={metrics['gini']:.4f}")
print(f"DPG: {metrics['dpg']}")
print("\n✅ Fair XGBoost trained and saved!")

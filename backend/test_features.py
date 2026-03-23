import pandas as pd
from data.loader import load_all_tables
from features.transaction_regularity import compute_transaction_regularity
from features.payment_behaviour import compute_payment_behaviour_score
from features.income_stability import compute_income_stability_index
from features.digital_footprint import compute_digital_footprint_score
from features.geo_income_index import compute_geo_income_index
from features.feature_pipeline import compute_all_features, compute_composite_behaviour_score, get_feature_summary

print("Loading tables...")
tables = load_all_tables()

print("\n--- Testing TRS ---")
trs = compute_transaction_regularity(tables["installments"])
print(trs.describe().round(4))

print("\n--- Testing PBS ---")
pbs = compute_payment_behaviour_score(tables["installments"])
print(pbs.describe().round(4))

print("\n--- Testing ISI ---")
isi = compute_income_stability_index(tables["pos_cash"], tables["credit_card"])
print(isi.describe().round(4))

print("\n--- Testing DFS ---")
dfs = compute_digital_footprint_score(tables["credit_card"], tables["pos_cash"], tables["bureau"])
print(dfs.describe().round(4))

print("\n--- Testing GII ---")
gii = compute_geo_income_index(tables["application_train"])
print(gii.describe().round(4))

print("\n--- Testing Full Feature Pipeline ---")
feature_df = compute_all_features(tables)
print(feature_df.describe().round(4))

print("\n--- Composite Behaviour Score ---")
cbs = compute_composite_behaviour_score(feature_df)
print(cbs.describe().round(4))

print("\n--- Feature Summary ---")
summary = get_feature_summary(feature_df)
print(summary.round(4))

print("\n✅ All features computed successfully on real data!")

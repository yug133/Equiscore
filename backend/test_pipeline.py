from data.loader import load_all_tables
from data.preprocessor import preprocess_application
from data.splitter import run_full_split_pipeline

print("Step 1: Loading tables...")
tables = load_all_tables()

print("\nStep 2: Preprocessing application_train...")
df = preprocess_application(tables["application_train"])

print("\nStep 3: Train-test split...")
X_train, X_test, y_train, y_test = run_full_split_pipeline(df)

print("\n✅ Pipeline test complete!")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train default rate: {y_train.mean():.4f}")
print(f"y_test  default rate: {y_test.mean():.4f}")
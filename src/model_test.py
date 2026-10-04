import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/raw/training_data.csv"

df = pd.read_csv(file_path)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

model_path = "models/bengali_ml_qa_model.pkl"

model = joblib.load(model_path)

# ==========================================
# 3. CREATE SAME TEST SET
# ==========================================

X = df["text"]
y = df["label"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 4. MODEL PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

confidence_scores = probabilities.max(axis=1)

# ==========================================
# 5. QA REPORT
# ==========================================

print("========== MODEL QA TEST REPORT ==========")

for text, expected, actual, confidence in zip(
    X_test,
    y_test,
    predictions,
    confidence_scores
):

    if expected == actual:
        result = "PASS"
    else:
        result = "FAIL"

    print("\n----------------------------------------")
    print(f"Input      : {text}")
    print(f"Expected   : {expected}")
    print(f"Actual     : {actual}")
    print(f"Confidence : {confidence:.2f}")
    print(f"Result     : {result}")

# ==========================================
# 6. SUMMARY
# ==========================================

total = len(y_test)
passed = sum(y_test == predictions)
failed = total - passed

print("\n========================================")
print("MODEL QA SUMMARY")
print("========================================")

print(f"Total Test Cases : {total}")
print(f"PASS             : {passed}")
print(f"FAIL             : {failed}")

if failed == 0:
    print("Overall Model QA : PASS")
else:
    print("Overall Model QA : FAIL")
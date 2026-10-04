import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. LOAD TRAINING DATA
# ==========================================

file_path = "data/raw/training_data.csv"

df = pd.read_csv(file_path)

print("========== ML MODEL TRAINING ==========")

print(f"\nTotal Records Loaded: {len(df)}")

# ==========================================
# 2. SPLIT DATA
# ==========================================

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Records: {len(X_train)}")
print(f"Testing Records: {len(X_test)}")

# ==========================================
# 3. CREATE ML PIPELINE
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

# ==========================================
# 4. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")

# ==========================================
# 5. TEST MODEL
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy:.2f}")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# ==========================================
# 6. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

import joblib

model_path = "models/bengali_ml_qa_model.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved successfully: {model_path}")

print("\n========== TRAINING COMPLETE ==========")
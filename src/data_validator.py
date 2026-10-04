import pandas as pd

# Dataset location
file_path = "data/raw/training_data.csv"

# Allowed labels
allowed_labels = {
    "VALID_BENGALI",
    "SPELLING_ERROR",
    "ROMANIZED_BENGALI",
    "MIXED_SCRIPT",
    "AMBIGUOUS"
}

# Read the CSV file
df = pd.read_csv(file_path)

print("========== DATA QA REPORT ==========")

# 1. Total records
total_records = len(df)
print(f"\nTotal Records: {total_records}")

# 2. Duplicate IDs
duplicate_ids = df["id"].duplicated().sum()
print(f"Duplicate IDs: {duplicate_ids}")

# 3. Empty or missing text
empty_text = df["text"].isna().sum() + (df["text"].astype(str).str.strip() == "").sum()
print(f"Empty Text: {empty_text}")

# 4. Invalid labels
invalid_labels = (~df["label"].isin(allowed_labels)).sum()
print(f"Invalid Labels: {invalid_labels}")

# 5. Label distribution
print("\nLabel Distribution:")

label_counts = df["label"].value_counts()

for label in allowed_labels:
    count = label_counts.get(label, 0)
    print(f"{label}: {count}")

# Overall result
if duplicate_ids == 0 and empty_text == 0 and invalid_labels == 0:
    print("\nOverall Data QA: PASS")
else:
    print("\nOverall Data QA: FAIL")
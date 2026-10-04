# Bengali ML QA Framework

An end-to-end Machine Learning Quality Assurance simulation project for Bengali text classification.

This project simulates a real-world ML QA workflow where a QA Engineer validates Bengali-language data, manually annotates test cases, evaluates model predictions, reviews model failures, checks quality metrics, performs SQL-based data QA, and makes a final model acceptance decision.

---

## 🎯 Project Objective

The objective of this project is to simulate the practical workflow of an ML QA Engineer working on a Bengali text classification system.

The project focuses on:

- Bengali script validation
- Bengali spelling-error classification
- Romanized Bengali / transliteration handling
- Mixed-script detection
- Ambiguous Bengali input handling
- Labelled test-set validation
- Model-output evaluation
- Human review of model predictions
- Precision, Recall and F1 evaluation
- Borderline / low-confidence case identification
- Large-scale data validation
- SQL-based QA checks
- Final model acceptance decision

---

## 🧪 QA Workflow

The project follows this workflow:

1. Prepare training data
2. Perform data quality validation
3. Train Bengali text classification model
4. Configure QA labels
5. Import labelled QA test data
6. Manually annotate test records
7. Run model evaluation
8. Compare model predictions with QA ground truth
9. Review model failures and low-confidence cases
10. Calculate Accuracy, Precision, Recall and F1
11. Perform SQL-based data QA
12. Record final acceptance decision
13. Export QA reports

---

## 🏷️ Classification Labels

The project uses five classification categories:

| Label | Description |
|---|---|
| VALID_BENGALI | Correct Bengali script/text |
| SPELLING_ERROR | Bengali text containing spelling errors |
| ROMANIZED_BENGALI | Bengali language written using Roman/Latin characters |
| MIXED_SCRIPT | Input containing both Bengali and Roman scripts |
| AMBIGUOUS | Short/context-dependent input requiring additional interpretation |

---

## 📂 Project Structure

```text
bengali-ml-qa-framework/
│
├── data/
│   ├── processed/
│   └── raw/
│       ├── training_data.csv
│       └── qa_test_data.csv
│
├── models/
│   └── bengali_ml_qa_model.pkl
│
├── reports/
│
├── src/
│   ├── data_validator.py
│   ├── model_test.py
│   ├── review_tool.py
│   └── train_model.py
│
├── .gitignore
└── README.md

🔍 Data QA

Before model training, the training dataset was validated for:

Total record count
Duplicate IDs
Empty text
Invalid labels
Label distribution

The training dataset contained 60 records across five classification categories.

The initial data-quality validation passed with no duplicate IDs, empty text or invalid labels.

🤖 ML Model

A Bengali text classification model was developed using:

Character-level TF-IDF features
Logistic Regression classifier
Scikit-learn Pipeline
Joblib model persistence

The model was intentionally evaluated as a QA subject rather than assuming that a trained model is production-ready.

📝 Manual Annotation

A Streamlit-based review interface was created to simulate a lightweight annotation/review tool.

The QA workflow allows the tester to:

Create a project
Configure classification labels
Import test data
Review records individually
Select the expected classification
Add QA comments
Save annotations
Navigate between records
Track annotation progress

A total of 25 QA test records were manually annotated.

📊 Model Evaluation

The manually annotated QA dataset was used as the ground truth for model evaluation.

Evaluation Results
Metric	Result
Accuracy	52.00%
Macro Precision	45.88%
Macro Recall	52.00%
Macro F1	44.09%
Key QA Finding

The model demonstrated a systematic classification weakness.

All five MIXED_SCRIPT test cases were incorrectly predicted as VALID_BENGALI.

All five SPELLING_ERROR test cases were also incorrectly predicted as VALID_BENGALI.

This indicates a consistent category-level model weakness rather than isolated random failures.

👤 Human Review

Model failures were sent through a human QA review workflow.

The review process included:

Model prediction inspection
Confidence-score inspection
QA decision
Review result classification
QA comments
Review history

12 model failures were manually reviewed and confirmed as incorrect.

The project also identifies low-confidence predictions for human review.

🗄️ SQL Data QA

A SQLite database is used to support QA data operations.

SQL-based validation can be used to investigate:

Record counts
Duplicate records
Invalid labels
Missing/empty text
Review status
Model results
QA investigation history

Example QA query:

SELECT COUNT(*)
FROM test_cases
WHERE result = 'FAIL';

Example label-distribution query:

SELECT expected_label, COUNT(*)
FROM test_cases
GROUP BY expected_label;
📋 Final Acceptance Decision

Based on the model evaluation and human review results, the model was:

REJECTED / SENT BACK

Acceptance Reason

The model showed systematic classification errors in the MIXED_SCRIPT and SPELLING_ERROR categories.

With:

52.00% Accuracy
45.88% Macro Precision
52.00% Macro Recall
44.09% Macro F1

the model requires improvement and re-evaluation before acceptance.

🛠️ Technologies Used
Python
Pandas
Scikit-learn
Streamlit
SQLite
SQL
Excel / CSV
Git / GitHub
VS Code
💡 QA Skills Demonstrated

This project demonstrates practical understanding of:

ML QA workflow
Bengali language QA
Transliteration / Romanized Bengali validation
Spelling-error validation
Mixed-script validation
Test-data preparation
Manual annotation
Model-output validation
Ground-truth comparison
Confidence-based review
Human-in-the-loop review
Accuracy / Precision / Recall / F1
Confusion-matrix analysis
SQL data validation
Defect pattern identification
Model acceptance / rejection decisions
QA reporting
⚠️ Project Scope

This is a self-developed ML QA simulation/portfolio project created to demonstrate hands-on understanding of an end-to-end ML QA workflow.

It is not presented as production ML QA experience from a previous employer.

🚀 Future Improvements

Planned improvements include:

Larger Bengali labelled datasets
Improved model architecture
Better transliteration coverage
More spelling-error variations
Advanced borderline-case handling
Regression test suite
Expanded SQL QA dashboards
Automated defect reporting
Model version comparison
Precision/Recall acceptance thresholds
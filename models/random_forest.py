import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==============================
# SETTINGS
# ==============================

INPUT_PATH = "data/final_with_anomalies.csv"
OUTPUT_RESULTS = "data/results_random_forest.csv"

FEATURES = [
    "rate",
    "duration_min",
    "abs_rate_change",
    "time_since_prev_min"
]

RANDOM_SEED = 42

print("Step 9: Random Forest Training")

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("\nInitial shape:", df.shape)

print("\nAnomaly label distribution:")
print(df["anomaly_label"].value_counts())

# ==============================
# PREPARE DATA
# ==============================

df = df.dropna(subset=FEATURES + ["anomaly_label"]).copy()

X = df[FEATURES]
y = df["anomaly_label"]

# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y   # εδώ ΘΕΛΟΥΜΕ stratify
)

# ==============================
# (OPTIONAL) SCALING
# ==============================
# Δεν είναι απαραίτητο για Random Forest
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# MODEL
# ==============================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight="balanced",   # ΠΟΛΥ σημαντικό για imbalance
    random_state=RANDOM_SEED,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# ==============================
# PREDICTION
# ==============================

y_pred = model.predict(X_test)

# ==============================
# EVALUATION
# ==============================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

tn, fp, fn, tp = cm.ravel()

false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (fn + tp)

print("\nMetrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("False Positive Rate:", false_positive_rate)
print("False Negative Rate:", false_negative_rate)

# ==============================
# SAVE RESULTS
# ==============================

results = pd.DataFrame({
    "model": ["Random Forest"],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "false_positive_rate": [false_positive_rate],
    "false_negative_rate": [false_negative_rate],
    "tn": [tn],
    "fp": [fp],
    "fn": [fn],
    "tp": [tp]
})

results.to_csv(OUTPUT_RESULTS, index=False)

print(f"\nSaved results to: {OUTPUT_RESULTS}")
print("\nDONE")
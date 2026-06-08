import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
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
OUTPUT_RESULTS = "data/results_isolation_forest.csv"

FEATURES = [
    "rate",
    "duration_min",
    "abs_rate_change",
    "time_since_prev_min"
]

RANDOM_SEED = 42
CONTAMINATION = 0.04

print("Step 6: Isolation Forest Training")

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("\nInitial shape:", df.shape)

print("\nAnomaly label distribution:")
print(df["anomaly_label"].value_counts())

print("\nAnomaly type distribution:")
print(df["anomaly_type"].value_counts())

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
    random_state=RANDOM_SEED
)

# ==============================
# SCALE FEATURES
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# TRAIN ONLY ON NORMAL DATA
# ==============================

X_train_normal = X_train_scaled[y_train == 0]

print("\nTraining data:")
print("Total training rows:", len(X_train_scaled))
print("Normal training rows used:", len(X_train_normal))
print("Anomaly rows excluded from training:", sum(y_train == 1))

# ==============================
# ISOLATION FOREST MODEL
# ==============================

model = IsolationForest(
    n_estimators=100,
    contamination=CONTAMINATION,
    random_state=RANDOM_SEED
)

print("\nTraining Isolation Forest on normal data only...")
model.fit(X_train_normal)

# ==============================
# PREDICTION
# ==============================

y_pred_raw = model.predict(X_test_scaled)

# Isolation Forest output:
#  1  = normal
# -1  = anomaly
y_pred = np.where(y_pred_raw == -1, 1, 0)

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
    "model": ["Isolation Forest"],
    "contamination": [CONTAMINATION],
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
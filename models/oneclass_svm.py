import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
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
OUTPUT_RESULTS = "data/results_oneclass_svm.csv"

FEATURES = [
    "rate",
    "duration_min",
    "abs_rate_change",
    "time_since_prev_min"
]

RANDOM_SEED = 42
NU = 0.04

# IMPORTANT: One-Class SVM is very slow on huge datasets
MAX_TRAIN_NORMAL = 50000

print("Step 7: One-Class SVM Training")

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

print("\nTraining data before sampling:")
print("Total training rows:", len(X_train_scaled))
print("Normal training rows:", len(X_train_normal))
print("Anomaly rows excluded from training:", sum(y_train == 1))

# ==============================
# SAMPLING FOR ONE-CLASS SVM
# ==============================

if len(X_train_normal) > MAX_TRAIN_NORMAL:
    np.random.seed(RANDOM_SEED)

    sample_idx = np.random.choice(
        len(X_train_normal),
        size=MAX_TRAIN_NORMAL,
        replace=False
    )

    X_train_normal = X_train_normal[sample_idx]

print("\nTraining data after sampling:")
print("Normal rows used:", len(X_train_normal))

# ==============================
# ONE-CLASS SVM MODEL
# ==============================

model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=NU
)

print("\nTraining One-Class SVM...")
model.fit(X_train_normal)

# ==============================
# PREDICTION
# ==============================

print("\nPredicting...")

y_pred_raw = model.predict(X_test_scaled)

# One-Class SVM output:
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
    "model": ["One-Class SVM"],
    "nu": [NU],
    "max_train_normal": [MAX_TRAIN_NORMAL],
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
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


# ==============================
# SETTINGS
# ==============================
INPUT_PATH = "data/final_with_anomalies.csv"

MODEL_PATH = "data/random_forest_model.pkl"
SCALER_PATH = "data/random_forest_scaler.pkl"

OUTPUT_CSV = "data/results_by_attack_type.csv"
OUTPUT_FIG = "data/results_by_attack_type_recall_f1.png"

RANDOM_SEED = 42
TEST_SIZE = 0.20

FEATURES = [
    "rate",
    "duration_min",
    "abs_rate_change",
    "time_since_prev_min"
]

ATTACK_TYPES = [
    "rate_manipulation",
    "sudden_change",
    "time_gap_manipulation",
    "replay_attack"
]


print("Step 6: Analysis of Random Forest results by attack type...")


# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_PATH, low_memory=False)

required_cols = FEATURES + ["anomaly_label", "anomaly_type"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df = df.dropna(subset=required_cols).copy()

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=FEATURES).copy()

print("\nDataset shape:", df.shape)
print("\nAnomaly type distribution:")
print(df["anomaly_type"].value_counts())


# ==============================
# SAME TEST SET AS RANDOM FOREST
# ==============================
X = df[FEATURES]
y = df["anomaly_label"].astype(int)

_, X_test, _, y_test, _, test_idx = train_test_split(
    X,
    y,
    df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

test_df = df.loc[test_idx].copy().reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print("\nTest set anomaly type distribution:")
print(test_df["anomaly_type"].value_counts())


# ==============================
# LOAD SAVED MODEL AND SCALER
# ==============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)


# ==============================
# OVERALL CHECK
# ==============================
print("\nOverall Random Forest test results:")
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, y_pred, zero_division=0), 4))


# ==============================
# ATTACK TYPE ANALYSIS
# ==============================
results = []

for attack_type in ATTACK_TYPES:
    attack_mask = test_df["anomaly_type"] == attack_type

    total_attack_records = int(attack_mask.sum())
    detected = int(((attack_mask) & (y_pred == 1)).sum())
    missed = int(((attack_mask) & (y_pred == 0)).sum())

    recall = detected / total_attack_records if total_attack_records > 0 else 0
    fnr = missed / total_attack_records if total_attack_records > 0 else 0

    normal_or_attack_mask = test_df["anomaly_type"].isin(["normal", attack_type])

    y_true_attack = (
        test_df.loc[normal_or_attack_mask, "anomaly_type"] == attack_type
    ).astype(int)

    y_pred_attack = y_pred[normal_or_attack_mask.to_numpy()]

    precision = precision_score(y_true_attack, y_pred_attack, zero_division=0)
    f1 = f1_score(y_true_attack, y_pred_attack, zero_division=0)

    results.append({
        "attack_type": attack_type,
        "total_attack_records": total_attack_records,
        "detected": detected,
        "missed": missed,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_negative_rate": fnr
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="recall",
    ascending=False
).reset_index(drop=True)

print("\nResults by attack type:")
print(results_df)


# ==============================
# SAVE CSV
# ==============================
os.makedirs("data", exist_ok=True)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved results to: {OUTPUT_CSV}")


# ==============================
# SAVE FIGURE
# ==============================
plt.figure(figsize=(10, 5))

x = range(len(results_df))

plt.bar(
    [i - 0.2 for i in x],
    results_df["recall"],
    width=0.4,
    label="Recall"
)

plt.bar(
    [i + 0.2 for i in x],
    results_df["f1_score"],
    width=0.4,
    label="F1-score"
)

plt.xticks(
    x,
    results_df["attack_type"],
    rotation=20,
    ha="right"
)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FIG, dpi=300)
plt.show()

print(f"Saved figure to: {OUTPUT_FIG}")


# ==============================
# INTERPRETATION
# ==============================
best_attack = results_df.iloc[0]["attack_type"]
worst_attack = results_df.iloc[-1]["attack_type"]

print("\nInterpretation:")
print(f"Best detected attack type: {best_attack}")
print(f"Most difficult attack type: {worst_attack}")
print("DONE")
import pandas as pd
import numpy as np


# SETTINGS

INPUT_DATA_PATH = "data/final_with_anomalies.csv"
INPUT_THRESHOLDS_PATH = "data/thresholds_per_drug.csv"
OUTPUT_SUMMARY_PATH = "data/anomaly_validation_summary.csv"

print("Step 4B: Validating injected anomalies...")

df = pd.read_csv(INPUT_DATA_PATH)
thr_df = pd.read_csv(INPUT_THRESHOLDS_PATH)

# Parse datetimes
df["starttime"] = pd.to_datetime(
    df["starttime"], format="%d/%m/%Y %H:%M", errors="coerce"
)
df["endtime"] = pd.to_datetime(
    df["endtime"], format="%d/%m/%Y %H:%M", errors="coerce"
)

print("\nLoaded dataset shape:", df.shape)

print("\nMissing datetimes:")
print(df[["starttime", "endtime"]].isna().sum())


# THRESHOLD LOOKUP

thresholds = {}
for _, row in thr_df.iterrows():
    thresholds[int(row["itemid"])] = {
        "rate_upper": float(row["rate_upper_used"]),
        "change_threshold": float(row["change_threshold"]),
        "time_gap_threshold": float(row["time_gap_threshold"]),
    }

# Map thresholds into dataset
df["rate_upper"] = df["itemid"].map(lambda x: thresholds[int(x)]["rate_upper"])
df["change_threshold"] = df["itemid"].map(lambda x: thresholds[int(x)]["change_threshold"])
df["time_gap_threshold"] = df["itemid"].map(lambda x: thresholds[int(x)]["time_gap_threshold"])


# SETTING VALIDATION RULES

# 1. rate_manipulation -> rate > rate_upper
df["valid_rate_manipulation"] = (
    (df["anomaly_type"] == "rate_manipulation") &
    (df["rate"] > df["rate_upper"])
)

# 2. sudden_change -> abs_rate_change > change_threshold
df["valid_sudden_change"] = (
    (df["anomaly_type"] == "sudden_change") &
    (df["abs_rate_change"] > df["change_threshold"])
)

# 3. time_gap_manipulation -> time_since_prev_min > time_gap_threshold
df["valid_time_gap"] = (
    (df["anomaly_type"] == "time_gap_manipulation") &
    (df["time_since_prev_min"] > df["time_gap_threshold"])
)

# 4. replay_attack -> small gap and near-zero change
# εδώ βάζουμε πρακτικό rule:
# - μικρό gap <= 30 min
# - abs_rate_change πολύ μικρό ή 0
REPLAY_TIME_WINDOW_MIN = 30

df["valid_replay_attack"] = (
    (df["anomaly_type"] == "replay_attack") &
    (df["time_since_prev_min"] <= REPLAY_TIME_WINDOW_MIN) &
    (df["abs_rate_change"] <= 0.01)
)


# SUMMARY

summary_rows = []

for anomaly_type, valid_col in [
    ("rate_manipulation", "valid_rate_manipulation"),
    ("sudden_change", "valid_sudden_change"),
    ("time_gap_manipulation", "valid_time_gap"),
    ("replay_attack", "valid_replay_attack"),
]:
    subset = df[df["anomaly_type"] == anomaly_type]
    total = len(subset)
    valid = subset[valid_col].sum()
    invalid = total - valid
    pct_valid = (valid / total * 100) if total > 0 else 0

    summary_rows.append({
        "anomaly_type": anomaly_type,
        "total_rows": total,
        "valid_rows": int(valid),
        "invalid_rows": int(invalid),
        "percent_valid": round(pct_valid, 2)
    })

summary = pd.DataFrame(summary_rows)

print("\nValidation summary:")
print(summary)

summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

print(f"\nSaved summary to: {OUTPUT_SUMMARY_PATH}")


# SHOW SOME INVALID EXAMPLES

print("\nSample invalid rows per anomaly type:")

for anomaly_type, valid_col in [
    ("rate_manipulation", "valid_rate_manipulation"),
    ("sudden_change", "valid_sudden_change"),
    ("time_gap_manipulation", "valid_time_gap"),
    ("replay_attack", "valid_replay_attack"),
]:
    bad = df[(df["anomaly_type"] == anomaly_type) & (~df[valid_col])]
    print(f"\n{anomaly_type} - invalid examples: {len(bad)}")
    if len(bad) > 0:
        print(
            bad[
                [
                    "subject_id",
                    "hadm_id",
                    "itemid",
                    "starttime",
                    "endtime",
                    "rate",
                    "rate_change",
                    "abs_rate_change",
                    "time_since_prev_min",
                    "anomaly_type"
                ]
            ].head(5)
        )

print("\nDONE")
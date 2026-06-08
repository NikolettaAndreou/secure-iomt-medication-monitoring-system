import pandas as pd
import numpy as np

# SETTINGS
INPUT_STATS_PATH = "data/stats_per_drug.csv"
OUTPUT_THRESHOLDS_PATH = "data/thresholds_per_drug.csv"

DRUG_NAMES = {
    221906: "Norepinephrine",
    222168: "Propofol",
    223258: "Insulin"
}

print("Step 4A: Creating thresholds per drug...")

# LOAD DATA
df = pd.read_csv(INPUT_STATS_PATH, low_memory=False)

print("Input shape:", df.shape)
print("\nColumns found:")
print(df.columns.tolist())

required_cols = [
    "itemid",
    "q3_rate",
    "iqr",
    "max_rate",
    "mean_abs_rate_change",
    "std_abs_rate_change",
    "q3_time_between",
    "iqr_time_between"
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in stats file: {missing_cols}")

# CALCULATE THRESHOLDS
thresholds = df.copy()

# Drug names
thresholds["drug_name"] = thresholds["itemid"].map(DRUG_NAMES)

# 1. Rate thresholds (IQR-based)
thresholds["rate_upper_theoretical"] = thresholds["q3_rate"] + 1.5 * thresholds["iqr"]
thresholds["rate_severe_theoretical"] = thresholds["q3_rate"] + 3.0 * thresholds["iqr"]

# 2. Change threshold
thresholds["change_threshold"] = (
    thresholds["mean_abs_rate_change"] + 3.0 * thresholds["std_abs_rate_change"]
)

# 3. Time gap threshold (IQR-based)
thresholds["time_gap_threshold"] = (
    thresholds["q3_time_between"] + 1.5 * thresholds["iqr_time_between"]
)

# OPTIONAL: REALISTIC RATE LIMITS
thresholds["rate_upper_used"] = thresholds["rate_upper_theoretical"]
thresholds["rate_severe_used"] = thresholds["rate_severe_theoretical"]
thresholds["rate_threshold_note"] = "theoretical"

for idx, row in thresholds.iterrows():
    theoretical_upper = row["rate_upper_theoretical"]
    theoretical_severe = row["rate_severe_theoretical"]
    max_rate = row["max_rate"]

    # If the theoretical threshold is already realistic, keep it
    if theoretical_upper <= max_rate:
        continue

    # Otherwise use a more practical threshold for injection
    upper_used = max_rate * 0.85
    severe_used = max_rate * 0.95

    # Safety: do not let it fall below Q3-based reasonable values
    upper_used = max(upper_used, row["q3_rate"] * 1.10)
    severe_used = max(severe_used, upper_used * 1.10)

    thresholds.at[idx, "rate_upper_used"] = upper_used
    thresholds.at[idx, "rate_severe_used"] = severe_used
    thresholds.at[idx, "rate_threshold_note"] = "adjusted_for_realistic_injection"

# ROUNDING FOR READABILITY
round_cols = [
    "q3_rate",
    "iqr",
    "max_rate",
    "rate_upper_theoretical",
    "rate_severe_theoretical",
    "rate_upper_used",
    "rate_severe_used",
    "change_threshold",
    "q3_time_between",
    "iqr_time_between",
    "time_gap_threshold"
]

for col in round_cols:
    thresholds[col] = thresholds[col].round(6)

# FINAL COLUMNS
output = thresholds[
    [
        "itemid",
        "drug_name",
        "q3_rate",
        "iqr",
        "max_rate",
        "rate_upper_theoretical",
        "rate_severe_theoretical",
        "rate_upper_used",
        "rate_severe_used",
        "change_threshold",
        "q3_time_between",
        "iqr_time_between",
        "time_gap_threshold",
        "rate_threshold_note"
    ]
].copy()

print("\nThresholds per drug:")
print(output)

output.to_csv(OUTPUT_THRESHOLDS_PATH, index=False)

print(f"\nSaved thresholds to: {OUTPUT_THRESHOLDS_PATH}")
print("DONE")
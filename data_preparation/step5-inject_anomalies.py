import pandas as pd
import numpy as np

# SETTINGS
INPUT_FEATURES_PATH = "data/filtered_drugs_features.csv"
INPUT_THRESHOLDS_PATH = "data/thresholds_per_drug.csv"
OUTPUT_PATH = "data/final_with_anomalies.csv"

RANDOM_SEED = 42

RATE_MANIPULATION_FRAC = 0.01
SUDDEN_CHANGE_FRAC = 0.01
TIME_GAP_MANIPULATION_FRAC = 0.01
REPLAY_ATTACK_FRAC = 0.005

REPLAY_TIME_WINDOW_MIN = 30

np.random.seed(RANDOM_SEED)

print("Step 5: Injecting anomalies...")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_FEATURES_PATH, low_memory=False)
thresholds_df = pd.read_csv(INPUT_THRESHOLDS_PATH, low_memory=False)

print("\nInitial features shape:", df.shape)
print("Initial thresholds shape:", thresholds_df.shape)

print("\nSample raw datetime values:")
print(df[["starttime", "endtime"]].head(5))

# ==============================
# DATETIME PARSING
# ==============================
# IMPORTANT:
# Step 3 saves datetimes without a fixed export format.
# So here we use flexible parsing instead of a strict format string.
df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")

print("\nMissing after datetime parsing:")
print(df[["starttime", "endtime"]].isna().sum())

# Keep only valid datetime rows
df = df.dropna(subset=["starttime", "endtime"]).copy()

print("Shape after datetime cleaning:", df.shape)

# ==============================
# SORT DATA
# ==============================
df = df.sort_values(by=["subject_id", "itemid", "starttime"]).reset_index(drop=True)

# ==============================
# LABEL NORMAL DATA
# ==============================
df["anomaly_label"] = 0
df["anomaly_type"] = "normal"

# ==============================
# THRESHOLDS LOOKUP
# ==============================
thresholds = {}
for _, row in thresholds_df.iterrows():
    thresholds[int(row["itemid"])] = {
        "rate_upper": float(row["rate_upper_used"]),
        "rate_severe": float(row["rate_severe_used"]),
        "change": float(row["change_threshold"]),
        "gap": float(row["time_gap_threshold"]),
    }

print("\nThresholds loaded:")
for itemid, vals in thresholds.items():
    print(itemid, vals)

# ==============================
# PREVIOUS ROW LOOKUP
# ==============================
prev_idx_map = {}
for (_, _), group in df.groupby(["subject_id", "itemid"], sort=False):
    idxs = group.index.tolist()
    for i, idx in enumerate(idxs):
        prev_idx_map[idx] = None if i == 0 else idxs[i - 1]

# ==============================
# HELPERS
# ==============================
def get_prev_row(idx):
    prev_idx = prev_idx_map.get(idx)
    if prev_idx is None:
        return None
    return df.loc[prev_idx]


def get_duration_minutes(row):
    start = row["starttime"]
    end = row["endtime"]

    if pd.notna(start) and pd.notna(end):
        dur = (end - start).total_seconds() / 60.0
        if pd.notna(dur) and np.isfinite(dur) and dur > 0:
            return float(dur)

    if (
        "duration_min" in row
        and pd.notna(row["duration_min"])
        and np.isfinite(row["duration_min"])
        and row["duration_min"] > 0
    ):
        return float(row["duration_min"])

    return np.nan


def validate_times(row):
    return (
        pd.notna(row["starttime"])
        and pd.notna(row["endtime"])
        and isinstance(row["starttime"], pd.Timestamp)
        and isinstance(row["endtime"], pd.Timestamp)
        and row["endtime"] > row["starttime"]
    )


def recompute_features(row, prev):
    row["duration_min"] = get_duration_minutes(row)

    if prev is not None:
        row["rate_change"] = float(row["rate"]) - float(prev["rate"])
        row["abs_rate_change"] = abs(row["rate_change"])
        row["time_since_prev_min"] = (
            (row["starttime"] - prev["starttime"]).total_seconds() / 60.0
        )
    else:
        row["rate_change"] = np.nan
        row["abs_rate_change"] = np.nan
        row["time_since_prev_min"] = np.nan

    return row


def sample_indices(frac):
    n = int(len(df) * frac)
    n = min(n, len(df))
    if n <= 0:
        return np.array([], dtype=int)
    return np.random.choice(df.index, size=n, replace=False)


def make_row_dict(source_row):
    row = source_row.to_dict()

    # explicitly preserve Timestamp
    row["starttime"] = pd.Timestamp(source_row["starttime"])
    row["endtime"] = pd.Timestamp(source_row["endtime"])

    return row


def finalize_and_append(row, anomaly_type, anomaly_list):
    if not validate_times(row):
        return False

    row["anomaly_label"] = 1
    row["anomaly_type"] = anomaly_type
    anomaly_list.append(row)
    return True

# ==============================
# ANOMALY INJECTION
# ==============================
anomalies = []

# ------------------------------
# 1. RATE MANIPULATION
# ------------------------------
rate_count = 0
for idx in sample_indices(RATE_MANIPULATION_FRAC):
    source = df.loc[idx]
    prev = get_prev_row(idx)
    if prev is None:
        continue

    row = make_row_dict(source)
    thr = thresholds[int(row["itemid"])]

    row["rate"] = thr["rate_upper"] * np.random.uniform(1.2, 1.5)
    row = recompute_features(row, prev)

    if finalize_and_append(row, "rate_manipulation", anomalies):
        rate_count += 1

print("Rate anomalies:", rate_count)

# ------------------------------
# 2. SUDDEN CHANGE
# ------------------------------
sudden_count = 0
for idx in sample_indices(SUDDEN_CHANGE_FRAC):
    source = df.loc[idx]
    prev = get_prev_row(idx)
    if prev is None:
        continue

    row = make_row_dict(source)
    thr = thresholds[int(row["itemid"])]

    change = thr["change"] * np.random.uniform(1.2, 2.0)
    direction = np.random.choice([-1, 1])

    new_rate = float(prev["rate"]) + direction * change
    if new_rate <= 0:
        new_rate = float(prev["rate"]) + abs(change)

    row["rate"] = new_rate
    row = recompute_features(row, prev)

    if finalize_and_append(row, "sudden_change", anomalies):
        sudden_count += 1

print("Sudden anomalies:", sudden_count)

# ------------------------------
# 3. TIME GAP MANIPULATION
# ------------------------------
timegap_count = 0
for idx in sample_indices(TIME_GAP_MANIPULATION_FRAC):
    source = df.loc[idx]
    prev = get_prev_row(idx)
    if prev is None:
        continue

    row = make_row_dict(source)
    thr = thresholds[int(row["itemid"])]

    duration = get_duration_minutes(row)
    if pd.isna(duration) or not np.isfinite(duration) or duration <= 0:
        continue

    gap = thr["gap"] * np.random.uniform(1.2, 2.0)

    row["starttime"] = pd.Timestamp(prev["starttime"]) + pd.Timedelta(minutes=gap)
    row["endtime"] = row["starttime"] + pd.Timedelta(minutes=duration)
    row = recompute_features(row, prev)

    if finalize_and_append(row, "time_gap_manipulation", anomalies):
        timegap_count += 1

print("Time gap anomalies:", timegap_count)

# ------------------------------
# 4. REPLAY ATTACK
# ------------------------------
replay_count = 0
for idx in sample_indices(REPLAY_ATTACK_FRAC):
    prev = get_prev_row(idx)
    if prev is None:
        continue

    row = make_row_dict(prev)

    duration = get_duration_minutes(row)
    if pd.isna(duration) or not np.isfinite(duration) or duration <= 0:
        continue

    gap = np.random.uniform(1, REPLAY_TIME_WINDOW_MIN)

    row["starttime"] = pd.Timestamp(prev["starttime"]) + pd.Timedelta(minutes=gap)
    row["endtime"] = row["starttime"] + pd.Timedelta(minutes=duration)
    row = recompute_features(row, prev)

    if finalize_and_append(row, "replay_attack", anomalies):
        replay_count += 1

print("Replay anomalies:", replay_count)

# ==============================
# BUILD ANOMALY DATAFRAME
# ==============================
print("\nDEBUG:")
print("Anomalies list length:", len(anomalies))

anomaly_df = pd.DataFrame(anomalies)

print("Anomaly df shape:", anomaly_df.shape)
print("Anomaly df empty:", anomaly_df.empty)

if not anomaly_df.empty:
    anomaly_df["starttime"] = pd.to_datetime(anomaly_df["starttime"], errors="coerce")
    anomaly_df["endtime"] = pd.to_datetime(anomaly_df["endtime"], errors="coerce")

    anomaly_df = anomaly_df.dropna(subset=["starttime", "endtime"]).copy()
    anomaly_df = anomaly_df[anomaly_df["endtime"] > anomaly_df["starttime"]].copy()

    print("\nAnomaly counts inside anomaly_df:")
    print(anomaly_df["anomaly_type"].value_counts())
else:
    print("\nWARNING: anomaly_df is empty!")

# ==============================
# FINAL DATASET
# ==============================
print("\nFINAL CHECK BEFORE CONCAT:")
print("df shape:", df.shape)
print("anomaly_df shape:", anomaly_df.shape)

final_df = pd.concat([df, anomaly_df], ignore_index=True)

# Put anomalies first so they are visible immediately in Excel
final_df = final_df.sort_values(
    by=["anomaly_label", "anomaly_type", "subject_id", "itemid", "starttime"],
    ascending=[False, True, True, True, True]
).reset_index(drop=True)

print("\nFinal shape:", final_df.shape)

print("\nAnomaly counts in final_df:")
print(final_df["anomaly_type"].value_counts())

print("\nMissing datetimes before save:")
print(final_df[["starttime", "endtime"]].isna().sum())

# ==============================
# SAVE
# ==============================
final_df.to_csv(
    OUTPUT_PATH,
    index=False,
    date_format="%d/%m/%Y %H:%M"
)

print("\nSaved to:", OUTPUT_PATH)

# ==============================
# VERIFY SAVED FILE
# ==============================
df_check = pd.read_csv(OUTPUT_PATH)

print("\nCHECK SAVED FILE:")
print(df_check["anomaly_type"].value_counts())

print("\nDONE")
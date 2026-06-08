import pandas as pd

# SETTINGS
INPUT_PATH = "data/filtered_drugs_clean.csv"
OUTPUT_FEATURES = "data/filtered_drugs_features.csv"
OUTPUT_STATS = "data/stats_per_drug.csv"

print("Step 3: Feature engineering & statistical analysis...")

# LOAD DATA
df = pd.read_csv(INPUT_PATH, low_memory=False)

# Convert datetime
df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")

print("Initial shape:", df.shape)

# Drop rows with missing critical grouping/time columns just in case
df = df.dropna(subset=["subject_id", "hadm_id", "itemid", "starttime", "endtime", "rate"]).copy()

# SORT DATA
df = df.sort_values(by=["subject_id", "hadm_id", "itemid", "starttime"]).copy()

# GROUPING KEY
group_cols = ["subject_id", "hadm_id", "itemid"]

# FEATURE ENGINEERING

# Rate change within same patient + admission + drug
df["rate_change"] = df.groupby(group_cols)["rate"].diff()

# Absolute change
df["abs_rate_change"] = df["rate_change"].abs()

# Time between infusions within same patient + admission + drug
df["time_since_prev_min"] = (
    df.groupby(group_cols)["starttime"]
    .diff()
    .dt.total_seconds() / 60.0
)

# REMOVE FIRST ROWS (NaN from diff)
df = df.dropna(subset=["rate_change", "time_since_prev_min"]).copy()

print("After feature engineering:", df.shape)

# STATISTICAL ANALYSIS PER DRUG
stats = (
    df.groupby("itemid")
    .agg(
        count=("rate", "count"),

        # Rate stats
        mean_rate=("rate", "mean"),
        median_rate=("rate", "median"),
        std_rate=("rate", "std"),
        min_rate=("rate", "min"),
        max_rate=("rate", "max"),
        q1_rate=("rate", lambda x: x.quantile(0.25)),
        q3_rate=("rate", lambda x: x.quantile(0.75)),
        iqr=("rate", lambda x: x.quantile(0.75) - x.quantile(0.25)),

        # Duration stats
        mean_duration=("duration_min", "mean"),
        std_duration=("duration_min", "std"),
        min_duration=("duration_min", "min"),
        max_duration=("duration_min", "max"),

        # Behavior stats - rate change
        mean_abs_rate_change=("abs_rate_change", "mean"),
        std_abs_rate_change=("abs_rate_change", "std"),

        # Behavior stats - time gaps
        mean_time_between=("time_since_prev_min", "mean"),
        std_time_between=("time_since_prev_min", "std"),
        q1_time_between=("time_since_prev_min", lambda x: x.quantile(0.25)),
        q3_time_between=("time_since_prev_min", lambda x: x.quantile(0.75)),
        iqr_time_between=("time_since_prev_min", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    )
    .reset_index()
)

print("\nFinal dataset shape:", df.shape)

print("\nItemid distribution:")
print(df["itemid"].value_counts())

print("\nFeature sample:")
print(df.head(10))

print("\nStatistics per drug:")
print(stats)

# SAVE RESULTS
df.to_csv(OUTPUT_FEATURES, index=False)
stats.to_csv(OUTPUT_STATS, index=False)

print(f"\nSaved feature dataset to: {OUTPUT_FEATURES}")
print(f"Saved statistics to: {OUTPUT_STATS}")
print("DONE")
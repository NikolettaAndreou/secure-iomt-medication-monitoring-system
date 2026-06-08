import pandas as pd

# SETTINGS

INPUT_PATH = "data/filtered_3drugs_raw.csv"
OUTPUT_PATH = "data/filtered_3drugs_clean.csv"

print("Step 2: Cleaning filtered dataset...")


# LOAD DATA

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("Initial shape:", df.shape)

print("\nInitial missing values:")
print(df.isna().sum())

print("\nOrder category distribution BEFORE cleaning:")
print(df["ordercategoryname"].value_counts(dropna=False))


# REMOVE MISSING CRITICAL VALUES

df = df.dropna(subset=["rate", "starttime", "endtime"]).copy()
print("\nAfter dropping missing rate/starttime/endtime:", df.shape)


# CONVERT DATA TYPES

df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")


# REMOVE INVALID ROWS AFTER CONVERSION

df = df.dropna(subset=["starttime", "endtime", "rate"]).copy()
print("After type conversion and dropping invalid rows:", df.shape)


# KEEP ONLY LOGICAL TIMES

df = df[df["endtime"] > df["starttime"]].copy()
print("After keeping endtime > starttime:", df.shape)


# KEEP ONLY POSITIVE RATES

df = df[df["rate"] > 0].copy()
print("After keeping positive rates:", df.shape)


# KEEP ONLY DRIPS

df = df[df["ordercategoryname"] == "01-Drips"].copy()
print("After keeping only 01-Drips:", df.shape)


# CREATE DURATION IN MINUTES

df["duration_min"] = (
    (df["endtime"] - df["starttime"]).dt.total_seconds() / 60.0
)


# REMOVE INVALID DURATIONS

df = df[df["duration_min"] > 0].copy()
print("After keeping duration_min > 0:", df.shape)


# OPTIONAL: REMOVE EXTREME RATE OUTLIERS

if len(df) > 0:
    upper_limit = df["rate"].quantile(0.99)
    print(f"99th percentile of rate: {upper_limit}")
    df = df[df["rate"] <= upper_limit].copy()
    print("After removing extreme rate outliers:", df.shape)


# FINAL CHECKS

print("\nFinal shape after cleaning:", df.shape)

print("\nItemid distribution after cleaning:")
print(df["itemid"].value_counts())

print("\nOrder category distribution after cleaning:")
print(df["ordercategoryname"].value_counts(dropna=False))

print("\nMissing values after cleaning:")
print(df.isna().sum())

print("\nRate summary:")
print(df["rate"].describe())

print("\nDuration summary:")
print(df["duration_min"].describe())

print("\nFirst 10 rows:")
print(df.head(10))


# SAVE CLEAN DATASET

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved cleaned dataset to: {OUTPUT_PATH}")
print("DONE")
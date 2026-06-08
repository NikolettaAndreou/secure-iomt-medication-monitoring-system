import pandas as pd

# SETTINGS

INPUT_PATH = "data/inputevents.csv"
OUTPUT_PATH = "data/filtered_3drugs_raw.csv"

SELECTED_ITEMIDS = [221906, 222168, 223258]

USECOLS = [
    "subject_id",
    "hadm_id",
    "itemid",
    "starttime",
    "endtime",
    "rate",
    "rateuom",
    "ordercategoryname"
]

CHUNKSIZE = 200000

print("Step 1: Filtering selected drugs from inputevents...")

filtered_chunks = []
total_rows = 0
kept_rows = 0

chunks = pd.read_csv(
    INPUT_PATH,
    usecols=USECOLS,
    chunksize=CHUNKSIZE,
    low_memory=False,
    on_bad_lines="skip"
)

for i, chunk in enumerate(chunks, start=1):
    total_rows += len(chunk)

    chunk = chunk[chunk["itemid"].isin(SELECTED_ITEMIDS)]

    if not chunk.empty:
        kept_rows += len(chunk)
        filtered_chunks.append(chunk)

    print(f"Chunk {i}: total rows read = {total_rows:,}, kept = {kept_rows:,}")

if not filtered_chunks:
    raise ValueError("No rows found for the selected itemids.")

df = pd.concat(filtered_chunks, ignore_index=True)

print("\nFiltered dataset shape:", df.shape)
print("\nItemid distribution:")
print(df["itemid"].value_counts())

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved filtered raw dataset to: {OUTPUT_PATH}")
print("DONE")
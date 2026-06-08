import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ==============================
# PAGE SETTINGS
# ==============================
st.set_page_config(
    page_title="Medication Anomaly Detection Dashboard",
    layout="wide"
)

# ==============================
# PATHS
# ==============================
DATA_PATH = "data/final_with_anomalies.csv"
LIVE_DATA_PATH = "data/live_stream_simulation.csv"
MODEL_PATH = "data/random_forest_model.pkl"
SCALER_PATH = "data/random_forest_scaler.pkl"

FEATURES = [
    "rate",
    "duration_min",
    "abs_rate_change",
    "time_since_prev_min"
]

DRUG_NAMES = {
    221906: "Norepinephrine",
    222168: "Propofol",
    223258: "Insulin"
}

MAX_ROWS_DASHBOARD = 100000
MAX_ROWS_SCATTER = 3000

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce", dayfirst=True)
    df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce", dayfirst=True)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["drug_name"] = df["itemid"].map(DRUG_NAMES)

    if len(df) > MAX_ROWS_DASHBOARD:
        df = df.sample(MAX_ROWS_DASHBOARD, random_state=42)

    return df


@st.cache_data
def load_live_data():
    live_df = pd.read_csv(LIVE_DATA_PATH, low_memory=False)

    live_df["starttime"] = pd.to_datetime(live_df["starttime"], errors="coerce", dayfirst=True)
    live_df["endtime"] = pd.to_datetime(live_df["endtime"], errors="coerce", dayfirst=True)
    live_df["prev_starttime"] = pd.to_datetime(live_df["prev_starttime"], errors="coerce", dayfirst=True)

    live_df["rate"] = pd.to_numeric(live_df["rate"], errors="coerce")
    live_df["prev_rate"] = pd.to_numeric(live_df["prev_rate"], errors="coerce")

    live_df = live_df.dropna(
        subset=[
            "subject_id",
            "hadm_id",
            "itemid",
            "starttime",
            "endtime",
            "rate",
            "prev_rate",
            "prev_starttime",
            "anomaly_label",
            "anomaly_type"
        ]
    ).copy()

    live_df = live_df[live_df["endtime"] > live_df["starttime"]].copy()
    live_df["drug_name"] = live_df["itemid"].map(DRUG_NAMES)

    return live_df.reset_index(drop=True)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


df = load_data()
live_df = load_live_data()
model, scaler = load_model()

# ==============================
# TITLE
# ==============================
st.title("Smart Medication Anomaly Detection Dashboard")
st.write(
    "Dashboard για την οπτικοποίηση φαρμακευτικών εγχύσεων "
    "και την προσομοίωση ανίχνευσης ανωμαλιών σε πραγματικό χρόνο με Random Forest."
)

# ==============================
# SIDEBAR FILTERS
# ==============================
st.sidebar.header("Filters")

selected_drugs = st.sidebar.multiselect(
    "Drug",
    options=df["drug_name"].dropna().unique(),
    default=df["drug_name"].dropna().unique()
)

selected_types = st.sidebar.multiselect(
    "Anomaly Type",
    options=df["anomaly_type"].dropna().unique(),
    default=df["anomaly_type"].dropna().unique()
)

filtered_df = df[
    (df["drug_name"].isin(selected_drugs)) &
    (df["anomaly_type"].isin(selected_types))
].copy()

# ==============================
# KPIs
# ==============================
st.subheader("System Overview")

total_records = len(filtered_df)
normal_records = len(filtered_df[filtered_df["anomaly_label"] == 0])
anomaly_records = len(filtered_df[filtered_df["anomaly_label"] == 1])
anomaly_rate = (anomaly_records / total_records * 100) if total_records > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{total_records:,}")
col2.metric("Normal Records", f"{normal_records:,}")
col3.metric("Anomalies", f"{anomaly_records:,}")
col4.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

# ==============================
# ANOMALY DISTRIBUTION
# ==============================
st.subheader("Anomaly Distribution")

col1, col2 = st.columns(2)

with col1:
    anomaly_counts = filtered_df["anomaly_type"].value_counts().reset_index()
    anomaly_counts.columns = ["anomaly_type", "count"]

    fig1 = px.bar(
        anomaly_counts,
        x="anomaly_type",
        y="count",
        title="Records by Anomaly Type"
    )

    st.plotly_chart(fig1, use_container_width=True)

with col2:
    drug_counts = (
        filtered_df
        .groupby(["drug_name", "anomaly_type"])
        .size()
        .reset_index(name="count")
    )

    fig2 = px.bar(
        drug_counts,
        x="drug_name",
        y="count",
        color="anomaly_type",
        title="Records by Drug and Type"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==============================
# RATE OVER TIME
# ==============================
st.subheader("Rate Over Time")

if len(filtered_df) > MAX_ROWS_SCATTER:
    scatter_df = filtered_df.sample(MAX_ROWS_SCATTER, random_state=42)
else:
    scatter_df = filtered_df.copy()

scatter_df = scatter_df.sort_values("starttime")

fig3 = px.scatter(
    scatter_df,
    x="starttime",
    y="rate",
    color="anomaly_type",
    hover_data=[
        "subject_id",
        "hadm_id",
        "drug_name",
        "rate",
        "duration_min",
        "abs_rate_change",
        "time_since_prev_min"
    ],
    title="Medication Rate Over Time"
)

st.plotly_chart(fig3, use_container_width=True)

# ==============================
# FEATURE ANALYSIS
# ==============================
st.subheader("Feature Analysis")

col1, col2 = st.columns(2)

with col1:
    fig4 = px.box(
        filtered_df,
        x="anomaly_type",
        y="abs_rate_change",
        title="Absolute Rate Change by Type"
    )

    st.plotly_chart(fig4, use_container_width=True)

with col2:
    fig5 = px.box(
        filtered_df,
        x="anomaly_type",
        y="time_since_prev_min",
        title="Time Gap by Type"
    )

    st.plotly_chart(fig5, use_container_width=True)

# ==============================
# LIVE STREAM SIMULATION
# ==============================
st.subheader("Live Stream Simulation")

st.write(
    "Η ενότητα αυτή χρησιμοποιεί μόνο το test set. "
    "Κάθε εγγραφή δίνεται ως raw record μαζί με το απαραίτητο previous context "
    "και το dashboard υπολογίζει δυναμικά τα χαρακτηριστικά πριν γίνει η πρόβλεψη."
)

if "live_index" not in st.session_state:
    st.session_state.live_index = 0

if "processed_records" not in st.session_state:
    st.session_state.processed_records = []

col1, col2, col3 = st.columns(3)

col1.metric("Live Test Dataset Records", f"{len(live_df):,}")
col2.metric("Current Position", f"{st.session_state.live_index} / {len(live_df)}")
col3.metric("Processed Records", f"{len(st.session_state.processed_records):,}")

button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    process_next = st.button("Process Next Record")

with button_col2:
    process_next_anomaly = st.button("Process Next Anomaly")

with button_col3:
    reset_stream = st.button("Reset Stream")

if reset_stream:
    st.session_state.live_index = 0
    st.session_state.processed_records = []
    st.success("Live stream reset successfully.")

# ==============================
# LIVE FEATURE CALCULATION
# ==============================
def compute_live_features(row):
    duration_min = (
        row["endtime"] - row["starttime"]
    ).total_seconds() / 60.0

    rate_change = row["rate"] - row["prev_rate"]
    abs_rate_change = abs(rate_change)

    time_since_prev_min = (
        row["starttime"] - row["prev_starttime"]
    ).total_seconds() / 60.0

    input_df = pd.DataFrame([{
        "rate": row["rate"],
        "duration_min": duration_min,
        "abs_rate_change": abs_rate_change,
        "time_since_prev_min": time_since_prev_min
    }])

    return input_df, rate_change

# ==============================
# PROCESS LIVE ROW
# ==============================
def process_row(row, title="Incoming Record"):
    input_df, rate_change = compute_live_features(row)

    input_scaled = scaler.transform(input_df[FEATURES])
    prediction = model.predict(input_scaled)[0]

    actual_label = int(row["anomaly_label"])

    processed_row = input_df.copy()
    processed_row["prediction"] = prediction
    processed_row["actual_label"] = actual_label
    processed_row["anomaly_type"] = row["anomaly_type"]
    processed_row["subject_id"] = row["subject_id"]
    processed_row["hadm_id"] = row["hadm_id"]
    processed_row["itemid"] = row["itemid"]
    processed_row["drug_name"] = row["drug_name"]
    processed_row["starttime"] = row["starttime"]
    processed_row["endtime"] = row["endtime"]

    st.session_state.processed_records.append(processed_row)

    st.write(title)

    st.write("1. Raw Incoming Data")
    raw_df = pd.DataFrame([{
        "subject_id": row["subject_id"],
        "hadm_id": row["hadm_id"],
        "itemid": row["itemid"],
        "drug_name": row["drug_name"],
        "starttime": row["starttime"],
        "endtime": row["endtime"],
        "rate": row["rate"],
        "actual_label": actual_label,
        "anomaly_type": row["anomaly_type"]
    }])
    st.dataframe(raw_df, use_container_width=True)

    st.write("2. Previous Context Used for Calculations")
    previous_context_df = pd.DataFrame([{
        "subject_id": row["subject_id"],
        "hadm_id": row["hadm_id"],
        "itemid": row["itemid"],
        "drug_name": row["drug_name"],
        "prev_starttime": row["prev_starttime"],
        "prev_rate": row["prev_rate"]
    }])
    st.dataframe(previous_context_df, use_container_width=True)

    st.write("3. Live Calculated Model Inputs")
    st.dataframe(input_df, use_container_width=True)

    st.write("4. Model Calculation Steps")
    calculation_df = pd.DataFrame([
        {
            "Step": "Raw input received",
            "Calculation": "The system receives subject_id, itemid, starttime, endtime and rate."
        },
        {
            "Step": "Previous context received",
            "Calculation": "The system uses prev_rate and prev_starttime saved in the test stream."
        },
        {
            "Step": "Duration calculation",
            "Calculation": "duration_min = (endtime - starttime) in minutes."
        },
        {
            "Step": "Rate change calculation",
            "Calculation": "rate_change = current rate - prev_rate."
        },
        {
            "Step": "Absolute rate change",
            "Calculation": "abs_rate_change = absolute value of rate_change."
        },
        {
            "Step": "Time gap calculation",
            "Calculation": "time_since_prev_min = current starttime - prev_starttime."
        },
        {
            "Step": "Scaling",
            "Calculation": "The four model inputs are transformed using random_forest_scaler.pkl."
        },
        {
            "Step": "Random Forest prediction",
            "Calculation": "The scaled inputs are passed to random_forest_model.pkl."
        }
    ])
    st.dataframe(calculation_df, use_container_width=True)

    st.write("5. Final Outcome")

    if prediction == 1:
        st.error("Anomaly Detected")
    else:
        st.success("Normal Infusion")

    st.write(f"Actual Label: **{actual_label}**")
    st.write(f"Model Prediction: **{prediction}**")

    if prediction == actual_label:
        st.success("Prediction matches the actual label.")
    else:
        st.warning("Prediction does not match the actual label.")

    st.subheader("Model Explanation")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Input Features Used by the Model")
        st.write(f"Rate: **{input_df['rate'].iloc[0]:.4f}**")
        st.write(f"Duration: **{input_df['duration_min'].iloc[0]:.4f} min**")
        st.write(f"Rate Change: **{rate_change:.4f}**")
        st.write(f"Abs Rate Change: **{input_df['abs_rate_change'].iloc[0]:.4f}**")
        st.write(f"Time Gap: **{input_df['time_since_prev_min'].iloc[0]:.4f} min**")

    with col2:
        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig_importance = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            title="Random Forest Feature Importance"
        )

        st.plotly_chart(fig_importance, use_container_width=True)

# ==============================
# BUTTON ACTIONS
# ==============================
if process_next:
    if st.session_state.live_index < len(live_df):
        row = live_df.iloc[st.session_state.live_index]
        process_row(row, title="Incoming Record")
        st.session_state.live_index += 1
    else:
        st.warning("No more records in the live stream dataset.")

if process_next_anomaly:
    remaining_df = live_df.iloc[st.session_state.live_index:]
    anomaly_rows = remaining_df[remaining_df["anomaly_label"] == 1]

    if not anomaly_rows.empty:
        next_anomaly_index = anomaly_rows.index[0]
        row = live_df.loc[next_anomaly_index]
        process_row(row, title="Incoming Anomaly Record")
        st.session_state.live_index = next_anomaly_index + 1
    else:
        st.warning("No more anomaly records found.")

# ==============================
# PROCESSED STREAM HISTORY
# ==============================
if len(st.session_state.processed_records) > 0:
    st.subheader("Processed Live Records")

    history_df = pd.concat(st.session_state.processed_records, ignore_index=True)

    st.dataframe(
        history_df.tail(20),
        use_container_width=True
    )

    detected_live_anomalies = len(history_df[history_df["prediction"] == 1])
    live_normal = len(history_df[history_df["prediction"] == 0])

    col1, col2 = st.columns(2)

    col1.metric("Live Normal Predictions", live_normal)
    col2.metric("Live Anomaly Predictions", detected_live_anomalies)

# ==============================
# RANDOM SAMPLE TEST
# ==============================
st.subheader("Random Sample Test")

if st.button("Test Random Dataset Record"):
    valid_df = df.dropna(subset=FEATURES + ["anomaly_label"]).copy()
    sample = valid_df.sample(1)

    X_sample = sample[FEATURES]
    X_sample_scaled = scaler.transform(X_sample)

    prediction = model.predict(X_sample_scaled)[0]
    actual = int(sample["anomaly_label"].iloc[0])

    st.dataframe(
        sample[
            [
                "subject_id",
                "hadm_id",
                "drug_name",
                "rate",
                "duration_min",
                "abs_rate_change",
                "time_since_prev_min",
                "anomaly_type",
                "anomaly_label"
            ]
        ],
        use_container_width=True
    )

    st.write(f"Actual Label: **{actual}**")
    st.write(f"Model Prediction: **{prediction}**")

    if prediction == 1:
        st.error("Prediction: Anomaly")
    else:
        st.success("Prediction: Normal")

# ==============================
# ANOMALY TABLE
# ==============================
st.subheader("Detected Anomaly Records")

anomalies_only = filtered_df[filtered_df["anomaly_label"] == 1]

st.dataframe(
    anomalies_only[
        [
            "subject_id",
            "hadm_id",
            "drug_name",
            "starttime",
            "endtime",
            "rate",
            "duration_min",
            "rate_change",
            "abs_rate_change",
            "time_since_prev_min",
            "anomaly_type"
        ]
    ].head(100),
    use_container_width=True
)

# ==============================
# SYSTEM STATUS
# ==============================
st.subheader("System Status")

if anomaly_records > 0:
    st.error(f"Warning: {anomaly_records:,} anomalous records detected in the selected sample.")
else:
    st.success("No anomalies detected in the selected sample.")
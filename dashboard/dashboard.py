import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json
import hmac
import hashlib
import os
import sys
import time
import subprocess
from cryptography.fernet import Fernet

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
SECURITY_LOG_PATH = "data/live_security_log.csv"
THRESHOLDS_PATH = "data/thresholds_per_drug.csv"

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

# ==============================
# LANGUAGE / TRANSLATIONS
# ==============================
TRANSLATIONS = {
    "language": {"en": "Language", "el": "Γλώσσα"},
    "english": {"en": "English", "el": "Αγγλικά"},
    "greek": {"en": "Greek", "el": "Ελληνικά"},
    "filters": {"en": "Filters", "el": "Φίλτρα"},
    "drug": {"en": "Drug", "el": "Φάρμακο"},
    "injected_scenario": {"en": "Injected Scenario", "el": "Σενάριο Εισαγόμενης Ανωμαλίας"},
    "dashboard_title": {"en": "Smart Medication Anomaly Detection Dashboard", "el": "Έξυπνο Dashboard Ανίχνευσης Ανωμαλιών Φαρμακευτικής Χορήγησης"},
    "dashboard_intro": {
        "en": "Dashboard for the visualization of medication infusion data and real-time anomaly detection using a Random Forest model. Live records are processed through an encrypted transmission workflow with HMAC-SHA256 integrity verification before prediction. All patient and admission identifiers displayed in the dashboard are pseudonymized for privacy protection.",
        "el": "Dashboard για την οπτικοποίηση δεδομένων φαρμακευτικής έγχυσης και την ανίχνευση ανωμαλιών σε προσομοιωμένο πραγματικό χρόνο με χρήση μοντέλου Random Forest. Κάθε live εγγραφή επεξεργάζεται μέσω κρυπτογραφημένης ροής μετάδοσης και ελέγχου ακεραιότητας HMAC-SHA256 πριν από την πρόβλεψη. Όλα τα αναγνωριστικά ασθενών και νοσηλειών εμφανίζονται ψευδωνυμοποιημένα για προστασία της ιδιωτικότητας."
    },
    "system_overview": {"en": "System Overview", "el": "Επισκόπηση Συστήματος"},
    "total_records": {"en": "Total Records", "el": "Συνολικές Εγγραφές"},
    "normal_records": {"en": "Normal Records", "el": "Φυσιολογικές Εγγραφές"},
    "anomalies": {"en": "Anomalies", "el": "Ανωμαλίες"},
    "anomaly_rate": {"en": "Anomaly Rate", "el": "Ποσοστό Ανωμαλιών"},
    "dataset_anomaly_distribution": {"en": "Dataset Anomaly Distribution", "el": "Κατανομή Ανωμαλιών Dataset"},
    "records_by_scenario": {"en": "Records by Injected Scenario", "el": "Εγγραφές ανά Σενάριο Εισαγόμενης Ανωμαλίας"},
    "records_by_drug_scenario": {"en": "Records by Drug and Injected Scenario", "el": "Εγγραφές ανά Φάρμακο και Σενάριο Εισαγόμενης Ανωμαλίας"},
    "live_stream_simulation": {"en": "Live Stream Simulation", "el": "Προσομοίωση Live Ροής Δεδομένων"},
    "live_stream_text": {
        "en": "This section uses only records from the test set. Each incoming live record is simulated as an encrypted packet, verified using HMAC-SHA256 integrity validation, and only after successful verification is the payload decrypted and processed by the Random Forest model.",
        "el": "Η ενότητα αυτή χρησιμοποιεί μόνο εγγραφές από το test set. Κάθε εισερχόμενη live εγγραφή προσομοιώνεται ως κρυπτογραφημένο πακέτο, επαληθεύεται με έλεγχο ακεραιότητας HMAC-SHA256 και μόνο μετά από επιτυχή επαλήθευση αποκρυπτογραφείται και επεξεργάζεται από το μοντέλο Random Forest."
    },
    "simulate_tampering": {"en": "Simulate tampered encrypted packet", "el": "Προσομοίωση αλλοιωμένου κρυπτογραφημένου πακέτου"},
    "tampering_help": {"en": "If enabled, the encrypted payload is intentionally modified so that the HMAC integrity verification fails.", "el": "Αν ενεργοποιηθεί, το κρυπτογραφημένο payload αλλοιώνεται σκόπιμα ώστε να αποτύχει ο έλεγχος ακεραιότητας HMAC."},
    "live_test_records": {"en": "Live Test Dataset Records", "el": "Εγγραφές Live Test Dataset"},
    "current_position": {"en": "Current Position", "el": "Τρέχουσα Θέση"},
    "processed_records_metric": {"en": "Processed Records", "el": "Επεξεργασμένες Εγγραφές"},
    "process_next_record": {"en": "Process Next Record", "el": "Επεξεργασία Επόμενης Εγγραφής"},
    "auto_stream": {"en": "Automatic Live Stream", "el": "Αυτόματη Live Ροή"},
    "auto_stream_help": {"en": "When enabled, the dashboard automatically processes the next live record every few seconds.", "el": "Όταν ενεργοποιηθεί, το dashboard επεξεργάζεται αυτόματα την επόμενη live εγγραφή κάθε λίγα δευτερόλεπτα."},
    "stream_speed": {"en": "Stream speed (seconds)", "el": "Ταχύτητα ροής (δευτερόλεπτα)"},
    "stop_on_anomaly": {"en": "Stop automatically when an anomaly is detected", "el": "Αυτόματη παύση όταν εντοπιστεί ανωμαλία"},
    "stop_on_anomaly_help": {"en": "Useful for presentation: the stream pauses when the model detects an anomaly, so the result and XAI explanation remain visible.", "el": "Χρήσιμο για παρουσίαση: η ροή σταματά όταν το μοντέλο εντοπίσει ανωμαλία, ώστε να παραμένουν ορατά το αποτέλεσμα και η XAI επεξήγηση."},
    "auto_paused_anomaly": {"en": "Automatic stream paused because an anomaly was detected.", "el": "Η αυτόματη ροή σταμάτησε επειδή εντοπίστηκε ανωμαλία."},
    "continue_live_simulation": {"en": "Continue Live Simulation", "el": "Συνέχεια Live Προσομοίωσης"},
    "reset_stream": {"en": "Reset Stream", "el": "Επαναφορά Ροής"},
    "stream_reset_success": {"en": "Live stream reset successfully.", "el": "Η live ροή επαναφέρθηκε επιτυχώς."},
    "incoming_record": {"en": "Incoming Record", "el": "Εισερχόμενη Εγγραφή"},
    "security_layer": {"en": "0. Security Layer: AES Encryption + HMAC-SHA256", "el": "0. Επίπεδο Ασφάλειας: Κρυπτογράφηση AES + HMAC-SHA256"},
    "payload_size": {"en": "Original Payload Size (bytes)", "el": "Αρχικό Μέγεθος Payload (bytes)"},
    "tampering_simulation": {"en": "Tampering Simulation", "el": "Προσομοίωση Αλλοίωσης"},
    "enabled": {"en": "Enabled", "el": "Ενεργή"},
    "disabled": {"en": "Disabled", "el": "Ανενεργή"},
    "hmac_success": {"en": "HMAC verification successful. Payload integrity confirmed.", "el": "Η επαλήθευση HMAC ήταν επιτυχής. Η ακεραιότητα του payload επιβεβαιώθηκε."},
    "record_rejected": {"en": "The record was rejected and was not sent to the Random Forest model.", "el": "Η εγγραφή απορρίφθηκε και δεν στάλθηκε στο μοντέλο Random Forest."},
    "live_inputs": {"en": "1. Live Calculated Model Inputs", "el": "1. Υπολογισμένες Live Είσοδοι Μοντέλου"},
    "show_steps": {"en": "2. Show Model Calculation Steps", "el": "2. Εμφάνιση Βημάτων Υπολογισμού Μοντέλου"},
    "final_outcome": {"en": "3. Final Outcome", "el": "3. Τελικό Αποτέλεσμα"},
    "anomaly_detected": {"en": "Anomaly Detected", "el": "Εντοπίστηκε Ανωμαλία"},
    "normal_infusion": {"en": "Normal Infusion", "el": "Φυσιολογική Έγχυση"},
    "evaluation_label": {"en": "Evaluation Label", "el": "Πραγματική Ετικέτα Αξιολόγησης"},
    "simulated_attack_type": {"en": "Simulated Attack Type", "el": "Τύπος Προσομοιωμένης Επίθεσης"},
    "simulated_attack_note": {"en": "This value comes from the labeled test dataset and is used only for evaluation/demo purposes. It is not predicted by the model.", "el": "Η τιμή αυτή προέρχεται από το labeled test dataset και χρησιμοποιείται μόνο για αξιολόγηση/επίδειξη. Δεν προβλέπεται από το μοντέλο."},
    "model_prediction": {"en": "Model Prediction", "el": "Πρόβλεψη Μοντέλου"},
    "prediction_matches": {"en": "Prediction matches the actual label.", "el": "Η πρόβλεψη συμφωνεί με την πραγματική ετικέτα."},
    "prediction_not_matches": {"en": "Prediction does not match the actual label.", "el": "Η πρόβλεψη δεν συμφωνεί με την πραγματική ετικέτα."},
    "model_explanation": {"en": "Model Explanation / XAI", "el": "Επεξήγηση Μοντέλου / XAI"},
    "input_features": {"en": "Input Features Used by the Model", "el": "Χαρακτηριστικά Εισόδου που χρησιμοποιεί το Μοντέλο"},
    "rate": {"en": "Rate", "el": "Ρυθμός Έγχυσης"},
    "duration": {"en": "Duration", "el": "Διάρκεια"},
    "rate_change": {"en": "Rate Change", "el": "Μεταβολή Ρυθμού"},
    "abs_rate_change": {"en": "Abs Rate Change", "el": "Απόλυτη Μεταβολή Ρυθμού"},
    "time_gap": {"en": "Time Gap", "el": "Χρονικό Κενό"},
    "feature_importance": {"en": "Random Forest Feature Importance", "el": "Σημαντικότητα Χαρακτηριστικών Random Forest"},
    "prediction_explanation": {"en": "### Explanation of the Prediction", "el": "### Επεξήγηση της Πρόβλεψης"},
    "feature_analysis_limits": {"en": "### Feature Analysis with Normal Limits", "el": "### Ανάλυση Χαρακτηριστικών με Φυσιολογικά Όρια"},
    "feature_limits_text": {
        "en": "This table compares the current live values with the normal upper limits calculated for the specific medication. This helps explain why a record may deviate from normal infusion behavior.",
        "el": "Ο πίνακας συγκρίνει τις τρέχουσες live τιμές με τα φυσιολογικά ανώτερα όρια που υπολογίστηκαν για το συγκεκριμένο φάρμακο. Αυτό βοηθά στην ερμηνεία του λόγου για τον οποίο μία εγγραφή μπορεί να αποκλίνει από τη φυσιολογική συμπεριφορά έγχυσης."
    },
    "no_more_records": {"en": "No more records in the live stream dataset.", "el": "Δεν υπάρχουν άλλες εγγραφές στη live ροή δεδομένων."},
    "security_log": {"en": "Live Security Log", "el": "Live Αρχείο Καταγραφής Ασφάλειας"},
    "security_log_text": {
        "en": "The log stores the HMAC signature and the verification result for each live record. Raw subject_id and hadm_id values are neither displayed nor stored in the log; they are replaced with pseudonymized references. Decrypted data is used only temporarily for feature calculation and anomaly prediction.",
        "el": "Το log αποθηκεύει την υπογραφή HMAC και το αποτέλεσμα επαλήθευσης για κάθε live εγγραφή. Τα raw subject_id και hadm_id δεν εμφανίζονται ούτε αποθηκεύονται στο log, αλλά αντικαθίστανται από ψευδωνυμοποιημένες αναφορές. Τα αποκρυπτογραφημένα δεδομένα χρησιμοποιούνται μόνο προσωρινά για υπολογισμό χαρακτηριστικών και πρόβλεψη ανωμαλίας."
    },
    "download_security_log": {"en": "Download Live Security Log as CSV", "el": "Λήψη Live Security Log ως CSV"},
    "save_security_log": {"en": "Save Security Log to data/live_security_log.csv", "el": "Αποθήκευση Security Log στο data/live_security_log.csv"},
    "security_log_saved": {"en": "Security log saved to", "el": "Το security log αποθηκεύτηκε στο"},
    "processed_live_records": {"en": "Processed Live Records", "el": "Επεξεργασμένες Live Εγγραφές"},
    "live_normal_predictions": {"en": "Live Normal Predictions", "el": "Live Φυσιολογικές Προβλέψεις"},
    "live_anomaly_predictions": {"en": "Live Anomaly Predictions", "el": "Live Προβλέψεις Ανωμαλιών"},
    "system_status": {"en": "System Status", "el": "Κατάσταση Συστήματος"},
    "warning_anomalies": {"en": "Warning", "el": "Προειδοποίηση"},
    "anomalous_selected": {"en": "anomalous records detected in the selected sample.", "el": "ανώμαλες εγγραφές εντοπίστηκαν στο επιλεγμένο δείγμα."},
    "no_anomalies_selected": {"en": "No anomalies detected in the selected sample.", "el": "Δεν εντοπίστηκαν ανωμαλίες στο επιλεγμένο δείγμα."},
}


def t(key):
    lang_code = st.session_state.get("lang_code", "en")
    return TRANSLATIONS.get(key, {}).get(lang_code, key)


# Sidebar language selector must be after t() is defined.
st.sidebar.header("Language / Γλώσσα")
language_choice = st.sidebar.radio(
    "Language / Γλώσσα",
    options=["English", "Ελληνικά"],
    horizontal=True
)
st.session_state.lang_code = "el" if language_choice == "Ελληνικά" else "en"


# ==============================
# DASHBOARD PRIVACY HELPERS
# ==============================
ID_MASK_SECRET = "demo_dashboard_masking_secret"


def pseudonymize_id(value, prefix):
    raw = f"{ID_MASK_SECRET}:{prefix}:{value}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"


def add_masked_identifiers(display_df):
    display_df = display_df.copy()

    if "subject_id" in display_df.columns:
        display_df["patient_ref"] = display_df["subject_id"].apply(lambda x: pseudonymize_id(x, "PAT"))

    if "hadm_id" in display_df.columns:
        display_df["admission_ref"] = display_df["hadm_id"].apply(lambda x: pseudonymize_id(x, "ADM"))

    cols_to_drop = [col for col in ["subject_id", "hadm_id"] if col in display_df.columns]
    display_df = display_df.drop(columns=cols_to_drop)

    first_cols = [col for col in ["patient_ref", "admission_ref"] if col in display_df.columns]
    other_cols = [col for col in display_df.columns if col not in first_cols]

    return display_df[first_cols + other_cols]


# ==============================
# SECURITY LAYER FOR LIVE DATA
# ==============================
HMAC_SECRET_KEY = b"demo_hmac_secret_key_for_thesis"


@st.cache_resource
def get_fernet():
    encryption_key = Fernet.generate_key()
    return Fernet(encryption_key)


fernet = get_fernet()


def serialize_live_row(row):
    payload = {
        "subject_id": int(row["subject_id"]),
        "hadm_id": int(row["hadm_id"]),
        "itemid": int(row["itemid"]),
        "drug_name": str(row["drug_name"]),
        "starttime": str(row["starttime"]),
        "endtime": str(row["endtime"]),
        "rate": float(row["rate"]),
        "prev_rate": float(row["prev_rate"]),
        "prev_starttime": str(row["prev_starttime"]),
        "anomaly_label": int(row["anomaly_label"]),
        "anomaly_type": str(row["anomaly_type"])
    }

    return json.dumps(payload).encode("utf-8")


def encrypt_and_sign_payload(payload_bytes):
    encrypted_payload = fernet.encrypt(payload_bytes)

    signature = hmac.new(
        HMAC_SECRET_KEY,
        encrypted_payload,
        hashlib.sha256
    ).hexdigest()

    return encrypted_payload, signature


def verify_and_decrypt_payload(encrypted_payload, received_signature):
    expected_signature = hmac.new(
        HMAC_SECRET_KEY,
        encrypted_payload,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, received_signature):
        if st.session_state.get("lang_code", "en") == "el":
            raise ValueError("Η επαλήθευση HMAC απέτυχε. Τα live δεδομένα ενδέχεται να έχουν αλλοιωθεί.")
        raise ValueError("HMAC verification failed. The live data may have been modified.")

    decrypted_payload = fernet.decrypt(encrypted_payload)
    payload = json.loads(decrypted_payload.decode("utf-8"))

    secure_row = pd.Series(payload)

    secure_row["starttime"] = pd.to_datetime(secure_row["starttime"], errors="coerce")
    secure_row["endtime"] = pd.to_datetime(secure_row["endtime"], errors="coerce")
    secure_row["prev_starttime"] = pd.to_datetime(secure_row["prev_starttime"], errors="coerce")

    secure_row["rate"] = float(secure_row["rate"])
    secure_row["prev_rate"] = float(secure_row["prev_rate"])
    secure_row["anomaly_label"] = int(secure_row["anomaly_label"])

    return secure_row


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


@st.cache_data
def load_thresholds():
    thresholds_df = pd.read_csv(THRESHOLDS_PATH, low_memory=False)
    thresholds_df["itemid"] = thresholds_df["itemid"].astype(int)
    return thresholds_df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


df = load_data()
live_df = load_live_data()
thresholds_df = load_thresholds()
model, scaler = load_model()


# ==============================
# XAI / NORMAL LIMIT HELPERS
# ==============================
def get_threshold_row(itemid):
    match = thresholds_df[thresholds_df["itemid"] == int(itemid)]

    if match.empty:
        return None

    return match.iloc[0]


def get_status(value, threshold):
    lang = st.session_state.get("lang_code", "en")

    if threshold is None or pd.isna(threshold) or pd.isna(value):
        return "⚪ Δεν υπάρχει διαθέσιμο όριο" if lang == "el" else "⚪ No limit available"

    if value > threshold:
        return "🔴 Πάνω από το φυσιολογικό όριο" if lang == "el" else "🔴 Above normal limit"

    if value > threshold * 0.80:
        return "🟠 Κοντά στο φυσιολογικό όριο" if lang == "el" else "🟠 Near normal limit"

    return "🟢 Εντός φυσιολογικού εύρους" if lang == "el" else "🟢 Within normal range"


def get_deviation(value, threshold):
    lang = st.session_state.get("lang_code", "en")

    if threshold is None or pd.isna(threshold) or pd.isna(value):
        return "-"

    difference = value - threshold

    if difference > 0:
        return f"+{difference:.4f} πάνω από το όριο" if lang == "el" else f"+{difference:.4f} above limit"

    return f"{abs(difference):.4f} κάτω από το όριο" if lang == "el" else f"{abs(difference):.4f} below limit"


def get_feature_explanation(feature, value, threshold):
    lang = st.session_state.get("lang_code", "en")

    if threshold is None or pd.isna(threshold):
        return "Δεν έχει οριστεί threshold για αυτό το χαρακτηριστικό." if lang == "el" else "No threshold was defined for this feature."

    if pd.isna(value):
        return "Δεν υπάρχει διαθέσιμη τιμή για αυτό το χαρακτηριστικό." if lang == "el" else "No value available for this feature."

    if value > threshold:
        if feature == "rate":
            return "Ο ρυθμός έγχυσης είναι υψηλότερος από το φυσιολογικό ανώτερο όριο για το συγκεκριμένο φάρμακο." if lang == "el" else "The infusion rate is higher than the normal upper limit for this medication."
        if feature == "abs_rate_change":
            return "Η μεταβολή του ρυθμού έγχυσης είναι μεγαλύτερη από την αναμενόμενη φυσιολογική μεταβολή." if lang == "el" else "The change in infusion rate is larger than the expected normal change."
        if feature == "time_since_prev_min":
            return "Το χρονικό κενό από την προηγούμενη εγγραφή έγχυσης είναι ασυνήθιστα μεγάλο." if lang == "el" else "The time gap from the previous infusion record is unusually large."
        if feature == "duration_min":
            return "Η διάρκεια της έγχυσης είναι υψηλότερη από το φυσιολογικό ανώτερο όριο." if lang == "el" else "The infusion duration is higher than the normal upper limit."

    if value > threshold * 0.80:
        return "Η τιμή παραμένει εντός φυσιολογικού ορίου, αλλά βρίσκεται κοντά στο threshold." if lang == "el" else "The value is still within the normal limit, but it is close to the threshold."

    return "Η τιμή βρίσκεται εντός του αναμενόμενου φυσιολογικού εύρους." if lang == "el" else "The value is within the expected normal range."


def build_feature_limit_table(input_df, rate_change, itemid):
    lang = st.session_state.get("lang_code", "en")
    thr_row = get_threshold_row(itemid)

    if lang == "el":
        columns = {
            "feature": "Χαρακτηριστικό",
            "current": "Τρέχουσα Τιμή",
            "limit": "Φυσιολογικό Ανώτερο Όριο",
            "deviation": "Απόκλιση",
            "status": "Κατάσταση",
            "explanation": "Επεξήγηση"
        }
    else:
        columns = {
            "feature": "Feature",
            "current": "Current Value",
            "limit": "Normal Upper Limit",
            "deviation": "Deviation",
            "status": "Status",
            "explanation": "Explanation"
        }

    if thr_row is None:
        return pd.DataFrame([{
            columns["feature"]: "Δεν βρέθηκαν thresholds" if lang == "el" else "No thresholds found",
            columns["current"]: "-",
            columns["limit"]: "-",
            columns["deviation"]: "-",
            columns["status"]: "⚪ Άγνωστο" if lang == "el" else "⚪ Unknown",
            columns["explanation"]: "Δεν βρέθηκαν τιμές threshold για αυτό το φάρμακο." if lang == "el" else "No threshold values were found for this medication."
        }])

    rate_value = float(input_df["rate"].iloc[0])
    duration_value = float(input_df["duration_min"].iloc[0])
    abs_change_value = float(input_df["abs_rate_change"].iloc[0])
    time_gap_value = float(input_df["time_since_prev_min"].iloc[0])

    rate_threshold = float(thr_row["rate_upper_used"])
    change_threshold = float(thr_row["change_threshold"])
    gap_threshold = float(thr_row["time_gap_threshold"])

    no_fixed = "Δεν υπάρχει σταθερό threshold" if lang == "el" else "No fixed threshold"
    informational = "⚪ Πληροφοριακό" if lang == "el" else "⚪ Informational"
    duration_explanation = (
        "Η διάρκεια χρησιμοποιείται από το μοντέλο ως πληροφορία πλαισίου."
        if lang == "el"
        else "Duration is used by the model as contextual information."
    )

    table = pd.DataFrame([
        {
            columns["feature"]: "rate",
            columns["current"]: f"{rate_value:.4f}",
            columns["limit"]: f"{rate_threshold:.4f}",
            columns["deviation"]: get_deviation(rate_value, rate_threshold),
            columns["status"]: get_status(rate_value, rate_threshold),
            columns["explanation"]: get_feature_explanation("rate", rate_value, rate_threshold)
        },
        {
            columns["feature"]: "duration_min",
            columns["current"]: f"{duration_value:.4f} min",
            columns["limit"]: no_fixed,
            columns["deviation"]: "-",
            columns["status"]: informational,
            columns["explanation"]: duration_explanation
        },
        {
            columns["feature"]: "rate_change",
            columns["current"]: f"{rate_change:.4f}",
            columns["limit"]: f"±{change_threshold:.4f}",
            columns["deviation"]: get_deviation(abs(rate_change), change_threshold),
            columns["status"]: get_status(abs(rate_change), change_threshold),
            columns["explanation"]: get_feature_explanation("abs_rate_change", abs(rate_change), change_threshold)
        },
        {
            columns["feature"]: "abs_rate_change",
            columns["current"]: f"{abs_change_value:.4f}",
            columns["limit"]: f"{change_threshold:.4f}",
            columns["deviation"]: get_deviation(abs_change_value, change_threshold),
            columns["status"]: get_status(abs_change_value, change_threshold),
            columns["explanation"]: get_feature_explanation("abs_rate_change", abs_change_value, change_threshold)
        },
        {
            columns["feature"]: "time_since_prev_min",
            columns["current"]: f"{time_gap_value:.4f} min",
            columns["limit"]: f"{gap_threshold:.4f} min",
            columns["deviation"]: get_deviation(time_gap_value, gap_threshold),
            columns["status"]: get_status(time_gap_value, gap_threshold),
            columns["explanation"]: get_feature_explanation("time_since_prev_min", time_gap_value, gap_threshold)
        }
    ])

    return table


def build_main_reason(feature_limit_df):
    lang = st.session_state.get("lang_code", "en")
    status_col = "Κατάσταση" if lang == "el" else "Status"
    feature_col = "Χαρακτηριστικό" if lang == "el" else "Feature"

    abnormal_pattern = "Πάνω από το φυσιολογικό όριο" if lang == "el" else "Above normal limit"
    near_pattern = "Κοντά στο φυσιολογικό όριο" if lang == "el" else "Near normal limit"

    abnormal_rows = feature_limit_df[feature_limit_df[status_col].str.contains(abnormal_pattern, na=False)]
    near_rows = feature_limit_df[feature_limit_df[status_col].str.contains(near_pattern, na=False)]

    if not abnormal_rows.empty:
        features = ", ".join(abnormal_rows[feature_col].tolist())
        return f"Η ισχυρότερη rule-based ένδειξη είναι ότι τα ακόλουθα χαρακτηριστικά υπερβαίνουν το φυσιολογικό ανώτερο όριο: {features}." if lang == "el" else f"The strongest rule-based indication is that the following feature(s) exceed the normal upper limit: {features}."

    if not near_rows.empty:
        features = ", ".join(near_rows[feature_col].tolist())
        return f"Κανένα χαρακτηριστικό δεν υπερβαίνει το threshold, αλλά τα ακόλουθα χαρακτηριστικά βρίσκονται κοντά στο φυσιολογικό όριο: {features}." if lang == "el" else f"No feature exceeds its threshold, but the following feature(s) are close to the normal limit: {features}."

    return "Όλες οι threshold-based τιμές βρίσκονται εντός των αναμενόμενων φυσιολογικών ορίων. Η πρόβλεψη του μοντέλου βασίζεται στο συνολικό μοτίβο Random Forest και όχι μόνο σε ένα threshold." if lang == "el" else "All threshold-based values are within the expected normal limits. The model prediction is based on the combined Random Forest pattern, not only on one threshold."


# ==============================
# TITLE
# ==============================
st.title(t("dashboard_title"))
st.write(t("dashboard_intro"))

# ==============================
# SIDEBAR FILTERS
# ==============================
st.sidebar.header(t("filters"))

selected_drugs = st.sidebar.multiselect(
    t("drug"),
    options=df["drug_name"].dropna().unique(),
    default=df["drug_name"].dropna().unique()
)

selected_types = st.sidebar.multiselect(
    t("injected_scenario"),
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
st.subheader(t("system_overview"))

total_records = len(filtered_df)
normal_records = len(filtered_df[filtered_df["anomaly_label"] == 0])
anomaly_records = len(filtered_df[filtered_df["anomaly_label"] == 1])
anomaly_rate = (anomaly_records / total_records * 100) if total_records > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric(t("total_records"), f"{total_records:,}")
col2.metric(t("normal_records"), f"{normal_records:,}")
col3.metric(t("anomalies"), f"{anomaly_records:,}")
col4.metric(t("anomaly_rate"), f"{anomaly_rate:.2f}%")

# ==============================
# ANOMALY DISTRIBUTION
# ==============================
st.subheader(t("dataset_anomaly_distribution"))

col1, col2 = st.columns(2)

with col1:
    anomaly_counts = filtered_df["anomaly_type"].value_counts().reset_index()
    anomaly_counts.columns = ["anomaly_type", "count"]

    fig1 = px.bar(
        anomaly_counts,
        x="anomaly_type",
        y="count",
        title=t("records_by_scenario")
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
        title=t("records_by_drug_scenario")
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==============================
# LIVE STREAM SIMULATION
# ==============================
st.subheader(t("live_stream_simulation"))
st.write(t("live_stream_text"))

simulate_tampering = st.checkbox(
    t("simulate_tampering"),
    value=False,
    help=t("tampering_help")
)

if "live_index" not in st.session_state:
    st.session_state.live_index = 0

if "processed_records" not in st.session_state:
    st.session_state.processed_records = []

if "security_logs" not in st.session_state:
    st.session_state.security_logs = []

if "auto_paused" not in st.session_state:
    st.session_state.auto_paused = False

col1, col2, col3 = st.columns(3)
col1.metric(t("live_test_records"), f"{len(live_df):,}")
col2.metric(t("current_position"), f"{st.session_state.live_index} / {len(live_df)}")
col3.metric(t("processed_records_metric"), f"{len(st.session_state.processed_records):,}")

auto_col1, auto_col2 = st.columns([2, 1])

with auto_col1:
    auto_stream = st.checkbox(
        t("auto_stream"),
        value=False,
        help=t("auto_stream_help")
    )

continue_button_placeholder = auto_col2.empty()


def render_continue_button():
    """Render the continue button next to the Automatic Live Stream control."""
    if st.session_state.get("auto_paused", False):
        with continue_button_placeholder.container():
            st.warning(t("auto_paused_anomaly"))

            if st.button(t("continue_live_simulation"), key="continue_live_simulation_top"):
                st.session_state.auto_paused = False
                st.rerun()


stream_speed = st.slider(
    t("stream_speed"),
    min_value=1,
    max_value=5,
    value=2
)

stop_on_anomaly = st.checkbox(
    t("stop_on_anomaly"),
    value=True,
    help=t("stop_on_anomaly_help")
)

button_col1, button_col2 = st.columns(2)

with button_col1:
    process_next = st.button(t("process_next_record"))

with button_col2:
    reset_stream = st.button(t("reset_stream"))

if reset_stream:
    st.session_state.live_index = 0
    st.session_state.processed_records = []
    st.session_state.security_logs = []
    st.session_state.auto_paused = False
    st.success(t("stream_reset_success"))


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
# SECURITY LOG FILE WRITER
# ==============================
def append_security_log_to_csv(log_row):
    """Append one security log entry to data/live_security_log.csv immediately."""
    log_dir = os.path.dirname(SECURITY_LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_df = pd.DataFrame([log_row])
    file_exists = os.path.exists(SECURITY_LOG_PATH)

    log_df.to_csv(
        SECURITY_LOG_PATH,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig"
    )


# ==============================
# PROCESS LIVE ROW
# ==============================
def process_row(row, title=None):
    if title is None:
        title = t("incoming_record")
    st.write(title)

    payload_bytes = serialize_live_row(row)
    encrypted_payload, signature = encrypt_and_sign_payload(payload_bytes)

    transmitted_payload = encrypted_payload

    if simulate_tampering:
        transmitted_payload = encrypted_payload[:-1] + (
            b"0" if encrypted_payload[-1:] != b"0" else b"1"
        )

    st.write(t("security_layer"))

    security_df = pd.DataFrame([{
        t("payload_size"): len(payload_bytes),
        "HMAC-SHA256": signature,
        t("tampering_simulation"): t("enabled") if simulate_tampering else t("disabled")
    }])

    st.dataframe(security_df, use_container_width=True)

    verification_status = "failed"
    security_error = ""

    try:
        secure_row = verify_and_decrypt_payload(transmitted_payload, signature)
        verification_status = "success"
        st.success(t("hmac_success"))
    except Exception as e:
        security_error = str(e)

        security_log_row = {
            "timestamp": pd.Timestamp.now(),
            "patient_ref": pseudonymize_id(row["subject_id"], "PAT"),
            "admission_ref": pseudonymize_id(row["hadm_id"], "ADM"),
            "itemid": row["itemid"],
            "simulated_attack_type": str(row.get("anomaly_type", "unknown")),
            "encrypted_payload": transmitted_payload.decode("utf-8", errors="ignore"),
            "hmac_sha256": signature,
            "verification_status": verification_status,
            "tampering_simulation": simulate_tampering,
            "prediction": None,
            "error": security_error
        }

        st.session_state.security_logs.append(security_log_row)
        append_security_log_to_csv(security_log_row)

        st.error(security_error)
        st.warning(t("record_rejected"))
        return None

    input_df, rate_change = compute_live_features(secure_row)

    input_scaled = scaler.transform(input_df[FEATURES])
    prediction = model.predict(input_scaled)[0]
    actual_label = int(secure_row["anomaly_label"])

    security_log_row = {
        "timestamp": pd.Timestamp.now(),
        "patient_ref": pseudonymize_id(secure_row["subject_id"], "PAT"),
        "admission_ref": pseudonymize_id(secure_row["hadm_id"], "ADM"),
        "itemid": secure_row["itemid"],
        "simulated_attack_type": str(secure_row.get("anomaly_type", "unknown")),
        "encrypted_payload": transmitted_payload.decode("utf-8", errors="ignore"),
        "hmac_sha256": signature,
        "verification_status": verification_status,
        "tampering_simulation": simulate_tampering,
        "prediction": int(prediction),
        "error": ""
    }

    st.session_state.security_logs.append(security_log_row)
    append_security_log_to_csv(security_log_row)

    processed_row = input_df.copy()
    processed_row["timestamp"] = pd.Timestamp.now()
    processed_row["rate_change"] = rate_change
    processed_row["prediction"] = prediction
    processed_row["actual_label"] = actual_label
    processed_row["patient_ref"] = pseudonymize_id(secure_row["subject_id"], "PAT")
    processed_row["admission_ref"] = pseudonymize_id(secure_row["hadm_id"], "ADM")
    processed_row["itemid"] = secure_row["itemid"]
    processed_row["simulated_attack_type"] = str(secure_row.get("anomaly_type", "unknown"))
    processed_row["starttime"] = secure_row["starttime"]
    processed_row["endtime"] = secure_row["endtime"]

    st.session_state.processed_records.append(processed_row)

    st.write(t("live_inputs"))
    st.dataframe(input_df, use_container_width=True)

    with st.expander(t("show_steps")):
        if st.session_state.get("lang_code", "en") == "el":
            calculation_df = pd.DataFrame([
                {"Βήμα": "Δημιουργία live payload", "Υπολογισμός": "Το σύστημα μετατρέπει την εισερχόμενη live εγγραφή σε JSON payload."},
                {"Βήμα": "Κρυπτογράφηση AES", "Υπολογισμός": "Το payload κρυπτογραφείται πριν από την επεξεργασία στο dashboard."},
                {"Βήμα": "Επαλήθευση HMAC-SHA256", "Υπολογισμός": "Το dashboard ελέγχει ακεραιότητα και αυθεντικότητα. Αν αποτύχει η επαλήθευση, η πρόβλεψη μπλοκάρεται."},
                {"Βήμα": "Αποκρυπτογράφηση", "Υπολογισμός": "Μόνο τα επαληθευμένα δεδομένα αποκρυπτογραφούνται και χρησιμοποιούνται για υπολογισμό χαρακτηριστικών."},
                {"Βήμα": "Προηγούμενο πλαίσιο", "Υπολογισμός": "Το σύστημα χρησιμοποιεί τα prev_rate και prev_starttime που υπάρχουν στο test stream."},
                {"Βήμα": "Υπολογισμός διάρκειας", "Υπολογισμός": "duration_min = (endtime - starttime) σε λεπτά."},
                {"Βήμα": "Υπολογισμός μεταβολής ρυθμού", "Υπολογισμός": "rate_change = current rate - prev_rate."},
                {"Βήμα": "Απόλυτη μεταβολή ρυθμού", "Υπολογισμός": "abs_rate_change = απόλυτη τιμή του rate_change."},
                {"Βήμα": "Υπολογισμός χρονικού κενού", "Υπολογισμός": "time_since_prev_min = current starttime - prev_starttime."},
                {"Βήμα": "Scaling", "Υπολογισμός": "Οι τέσσερις είσοδοι του μοντέλου μετασχηματίζονται με το random_forest_scaler.pkl."},
                {"Βήμα": "Πρόβλεψη Random Forest", "Υπολογισμός": "Οι scaled είσοδοι περνούν στο random_forest_model.pkl."}
            ])
        else:
            calculation_df = pd.DataFrame([
                {"Step": "Live payload creation", "Calculation": "The system serializes the incoming live record as a JSON payload."},
                {"Step": "AES encryption", "Calculation": "The payload is encrypted before being processed by the dashboard."},
                {"Step": "HMAC-SHA256 verification", "Calculation": "The dashboard verifies integrity/authenticity. If verification fails, prediction is blocked."},
                {"Step": "Decryption", "Calculation": "Only verified data are decrypted and used for feature calculation."},
                {"Step": "Previous context received", "Calculation": "The system uses prev_rate and prev_starttime saved in the test stream."},
                {"Step": "Duration calculation", "Calculation": "duration_min = (endtime - starttime) in minutes."},
                {"Step": "Rate change calculation", "Calculation": "rate_change = current rate - prev_rate."},
                {"Step": "Absolute rate change", "Calculation": "abs_rate_change = absolute value of rate_change."},
                {"Step": "Time gap calculation", "Calculation": "time_since_prev_min = current starttime - prev_starttime."},
                {"Step": "Scaling", "Calculation": "The four model inputs are transformed using random_forest_scaler.pkl."},
                {"Step": "Random Forest prediction", "Calculation": "The scaled inputs are passed to random_forest_model.pkl."}
            ])

        st.dataframe(calculation_df, use_container_width=True)

    st.write(t("final_outcome"))

    if prediction == 1:
        st.error(t("anomaly_detected"))
    else:
        st.success(t("normal_infusion"))

    st.write(f"{t('evaluation_label')}: **{actual_label}**")
    st.write(f"{t('model_prediction')}: **{prediction}**")
    st.write(f"{t('simulated_attack_type')}: **{secure_row.get('anomaly_type', 'unknown')}**")
    st.caption(t("simulated_attack_note"))

    if prediction == actual_label:
        st.success(t("prediction_matches"))
    else:
        st.warning(t("prediction_not_matches"))

    # ==============================
    # MODEL EXPLANATION / XAI
    # ==============================
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    top_feature = importance_df.iloc[0]["Feature"]
    top_importance = importance_df.iloc[0]["Importance"]

    if st.session_state.get("lang_code", "en") == "el":
        feature_labels = {
            "rate": "ρυθμός έγχυσης",
            "duration_min": "διάρκεια έγχυσης",
            "abs_rate_change": "απόλυτη μεταβολή ρυθμού",
            "time_since_prev_min": "χρονικό κενό από την προηγούμενη εγγραφή"
        }
    else:
        feature_labels = {
            "rate": "infusion rate",
            "duration_min": "infusion duration",
            "abs_rate_change": "absolute rate change",
            "time_since_prev_min": "time gap from the previous record"
        }

    readable_top_feature = feature_labels.get(top_feature, top_feature)

    feature_limit_df = build_feature_limit_table(
        input_df=input_df,
        rate_change=rate_change,
        itemid=secure_row["itemid"]
    )

    main_reason = build_main_reason(feature_limit_df)

    col1, col2 = st.columns(2, vertical_alignment="top")

    with col1:
        st.subheader(t("model_explanation"))
        st.write(t("input_features"))
        st.write(f"{t('rate')}: **{input_df['rate'].iloc[0]:.4f}**")
        st.write(f"{t('duration')}: **{input_df['duration_min'].iloc[0]:.4f} min**")
        st.write(f"{t('rate_change')}: **{rate_change:.4f}**")
        st.write(f"{t('abs_rate_change')}: **{input_df['abs_rate_change'].iloc[0]:.4f}**")
        st.write(f"{t('time_gap')}: **{input_df['time_since_prev_min'].iloc[0]:.4f} min**")

    with col2:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        importance_chart_df = importance_df.copy()
        importance_chart_df["Feature"] = importance_chart_df["Feature"].map(
            lambda name: feature_labels.get(name, name)
        )

        fig_importance = px.bar(
            importance_chart_df,
            x="Feature",
            y="Importance",
            text="Importance",
        )
        chart_height = 290
        max_importance = float(importance_chart_df["Importance"].max())
        fig_importance.update_traces(
            marker_color="#2563eb",
            marker_line_width=0,
            texttemplate="%{y:.3f}",
            textposition="outside",
            cliponaxis=False,
            width=0.55,
        )
        fig_importance.update_layout(
            height=chart_height,
            margin=dict(t=44, b=36, l=48, r=16),
            bargap=0.12,
            title=dict(
                text=t("feature_importance"),
                x=0.5,
                xanchor="center",
                font=dict(size=14),
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(
                title="Importance",
                range=[0, max(max_importance * 1.22, 0.05)],
                tickfont=dict(size=11),
                gridcolor="rgba(0,0,0,0.08)",
            ),
            xaxis=dict(tickfont=dict(size=11)),
            uniformtext_minsize=9,
            showlegend=False,
        )
        fig_importance.update_xaxes(tickangle=0)
        fig_importance.update_yaxes(automargin=True)

        st.plotly_chart(
            fig_importance,
            use_container_width=True,
            height=chart_height,
            config={"displayModeBar": False},
        )

    st.markdown(t("prediction_explanation"))

    if st.session_state.get("lang_code", "en") == "el":
        if prediction == 1:
            st.warning(
                f"""
                Το μοντέλο ταξινόμησε αυτή την εγγραφή ως **ανωμαλία**.

                Αυτό σημαίνει ότι οι τιμές της τρέχουσας εγγραφής έγχυσης διαφέρουν από τα φυσιολογικά μοτίβα έγχυσης που έμαθε το μοντέλο κατά την εκπαίδευση.

                {main_reason}

                Με βάση τη συνολική σημαντικότητα χαρακτηριστικών του Random Forest, το πιο σημαντικότερο χαρακτηριστικό είναι το **{readable_top_feature}**, με score σημαντικότητας **{top_importance:.3f}**.
                """
            )
        else:
            st.info(
                f"""
                Το μοντέλο ταξινόμησε αυτή την εγγραφή ως **φυσιολογική έγχυση**.

                Αυτό σημαίνει ότι οι τιμές της τρέχουσας εγγραφής είναι συμβατές με τα φυσιολογικά μοτίβα έγχυσης που έμαθε το μοντέλο κατά την εκπαίδευση.

                {main_reason}

                Με βάση τη συνολική σημαντικότητα χαρακτηριστικών του Random Forest, το πιο επιδραστικό χαρακτηριστικό είναι το **{readable_top_feature}**, με score σημαντικότητας **{top_importance:.3f}**.
                """
            )
    else:
        if prediction == 1:
            st.warning(
                f"""
                The model classified this record as an **anomaly**.

                This means that the values of the current infusion record differ from the normal infusion patterns learned during training.

                {main_reason}

                Based on the overall Random Forest feature importance, the most influential feature is **{readable_top_feature}**, with an importance score of **{top_importance:.3f}**.
                """
            )
        else:
            st.info(
                f"""
                The model classified this record as a **normal infusion**.

                This means that the values of the current infusion record are consistent with the normal infusion patterns learned during training.

                {main_reason}

                Based on the overall Random Forest feature importance, the most influential feature is **{readable_top_feature}**, with an importance score of **{top_importance:.3f}**.
                """
            )

    st.markdown(t("feature_analysis_limits"))
    st.write(t("feature_limits_text"))
    st.dataframe(
        feature_limit_df,
        use_container_width=True,
        hide_index=True
    )

    return int(prediction)


# ==============================
# BUTTON / AUTO STREAM ACTIONS
# ==============================
def process_next_live_record():
    if st.session_state.live_index < len(live_df):
        row = live_df.iloc[st.session_state.live_index]
        prediction = process_row(row, title=t("incoming_record"))
        st.session_state.live_index += 1
        return prediction

    st.warning(t("no_more_records"))
    return None


if process_next:
    st.session_state.auto_paused = False
    process_next_live_record()

if auto_stream and not process_next and not st.session_state.get("auto_paused", False):
    prediction = process_next_live_record()

    if prediction == 1 and stop_on_anomaly:
        st.session_state.auto_paused = True
    elif st.session_state.live_index < len(live_df):
        time.sleep(stream_speed)
        st.rerun()
    else:
        st.warning(t("no_more_records"))

render_continue_button()

# ==============================
# SECURITY LOG FILE ACCESS
# ==============================
st.subheader(t("security_log"))
st.write(t("security_log_text"))

st.info(
    f"Security logs are saved automatically to: {SECURITY_LOG_PATH}"
    if st.session_state.get("lang_code", "en") == "en"
    else f"Τα security logs αποθηκεύονται αυτόματα στο αρχείο: {SECURITY_LOG_PATH}"
)

if st.button(
    "Open Security Log File"
    if st.session_state.get("lang_code", "en") == "en"
    else "Άνοιγμα αρχείου Security Logs"
):
    if os.path.exists(SECURITY_LOG_PATH):
        try:
            absolute_log_path = os.path.abspath(SECURITY_LOG_PATH)

            if os.name == "nt":
                os.startfile(absolute_log_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", absolute_log_path])
            else:
                subprocess.Popen(["xdg-open", absolute_log_path])

            st.success(
                "Security log file opened."
                if st.session_state.get("lang_code", "en") == "en"
                else "Το αρχείο των security logs άνοιξε."
            )
        except Exception as e:
            st.error(
                f"Could not open the log file automatically. File path: {os.path.abspath(SECURITY_LOG_PATH)}. Error: {e}"
                if st.session_state.get("lang_code", "en") == "en"
                else f"Δεν ήταν δυνατό να ανοίξει αυτόματα το αρχείο. Διαδρομή αρχείου: {os.path.abspath(SECURITY_LOG_PATH)}. Σφάλμα: {e}"
            )
    else:
        st.warning(
            "No security log file exists yet. Process a live record first."
            if st.session_state.get("lang_code", "en") == "en"
            else "Δεν υπάρχει ακόμη αρχείο security logs. Επεξεργάσου πρώτα μία live εγγραφή."
        )

# ==============================
# SYSTEM STATUS
# ==============================
st.subheader(t("system_status"))

if anomaly_records > 0:
    st.error(f"{t('warning_anomalies')}: {anomaly_records:,} {t('anomalous_selected')}")
else:
    st.success(t("no_anomalies_selected"))
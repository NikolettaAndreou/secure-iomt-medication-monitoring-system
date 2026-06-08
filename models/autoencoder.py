import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# SETTINGS

INPUT_PATH = "data/final_with_anomalies.csv"
OUTPUT_RESULTS = "data/results_autoencoder.csv"

FEATURES = [
    "rate",
    "duration_min",
    "abs_rate_change",
    "time_since_prev_min"
]

RANDOM_SEED = 42
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.001
THRESHOLD_PERCENTILE = 96

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Step 8: Autoencoder Training")


# LOAD DATA


df = pd.read_csv(INPUT_PATH, low_memory=False)

print("\nInitial shape:", df.shape)

print("\nAnomaly label distribution:")
print(df["anomaly_label"].value_counts())

print("\nAnomaly type distribution:")
print(df["anomaly_type"].value_counts())


# PREPARE DATA

df = df.dropna(subset=FEATURES + ["anomaly_label"]).copy()

X = df[FEATURES]
y = df["anomaly_label"]


# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED
)


# SCALE FEATURES

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# TRAIN ONLY ON NORMAL DATA

X_train_normal = X_train_scaled[y_train == 0]

print("\nTraining data:")
print("Total training rows:", len(X_train_scaled))
print("Normal training rows used:", len(X_train_normal))
print("Anomaly rows excluded from training:", sum(y_train == 1))


# CONVERT TO TORCH TENSORS


X_train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# AUTOENCODER MODEL


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


input_dim = len(FEATURES)

model = Autoencoder(input_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# TRAINING


print("\nTraining Autoencoder...")

for epoch in range(EPOCHS):
    total_loss = 0

    for batch_x, _ in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

print("Training completed.")


# RECONSTRUCTION ERROR


model.eval()

with torch.no_grad():
    train_reconstructed = model(X_train_tensor)
    train_errors = torch.mean((X_train_tensor - train_reconstructed) ** 2, dim=1).numpy()

    test_reconstructed = model(X_test_tensor)
    test_errors = torch.mean((X_test_tensor - test_reconstructed) ** 2, dim=1).numpy()



# THRESHOLD

threshold = np.percentile(train_errors, THRESHOLD_PERCENTILE)

print("\nReconstruction error threshold:", threshold)

# If reconstruction error > threshold => anomaly
y_pred = np.where(test_errors > threshold, 1, 0)



# EVALUATION

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


# SAVING

results = pd.DataFrame({
    "model": ["Autoencoder"],
    "epochs": [EPOCHS],
    "batch_size": [BATCH_SIZE],
    "learning_rate": [LEARNING_RATE],
    "threshold_percentile": [THRESHOLD_PERCENTILE],
    "threshold": [threshold],
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
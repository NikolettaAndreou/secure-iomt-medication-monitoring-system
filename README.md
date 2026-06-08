# Secure IoMT Medication Monitoring System

**Bachelor Thesis Project – Applied Computer Science**

## Overview

This project presents a secure Internet of Medical Things (IoMT) medication monitoring system that combines machine learning, cybersecurity mechanisms, and Explainable Artificial Intelligence (XAI) to detect anomalous medication infusion events.

The system was developed using clinical medication administration records from the MIMIC-IV database and evaluated under multiple simulated cyberattack scenarios targeting connected medical devices.

---

## System Architecture

<p align="center">
  <img src="diagrams/Architecture%20diagram.png" width="500">
</p>

The solution integrates data processing, cybersecurity controls, machine learning-based anomaly detection, explainability mechanisms, and an interactive monitoring dashboard.

---

## Threat Model

<p align="center">
  <img src="diagrams/Threat%20Model%20Diagram.png" width="500">
</p>

The system is designed to detect and mitigate common healthcare IoMT threats, including spoofing, data tampering, rate manipulation, time-gap manipulation, and replay attacks.

---

## Key Features

* Machine learning-based anomaly detection
* Explainable AI (XAI)
* AES encryption for data confidentiality
* HMAC-SHA256 integrity verification
* Replay attack detection
* Continuous data stream simulation
* Interactive Streamlit dashboard
* Security event logging

---

## Simulated Attack Scenarios

Four cyberattack scenarios were injected into the dataset:

1. Rate Manipulation
2. Sudden Rate Change
3. Time Gap Manipulation
4. Replay Attack

---

## Machine Learning Models Evaluated

* Random Forest
* Logistic Regression
* Isolation Forest
* One-Class SVM
* Autoencoder

---

## Results

| Metric              | Random Forest |
| ------------------- | ------------- |
| Accuracy            | 95.75%        |
| Recall              | 97.9%         |
| False Negative Rate | 2.1%          |

The Random Forest model achieved the best overall performance and was selected as the final anomaly detection model.

### Accuracy Comparison

![Accuracy per Model](results/accuracy%20per%20model.png)

### False Positive and False Negative Rates

![FPR and FNR per Model](results/fpr%20and%20fnr%20per%20model.png)

### Recall and F1 Score Comparison

![Recall and F1 Score](results/recall%20and%20f1%20score.png)

### Performance by Attack Type

![Results by Attack Type](results/results_by_attack_type_recall_f1.png)

---

## Dashboard Workflow

<p align="center">
  <img src="diagrams/dashboard%20workflow-diagram.png" width="350">
</p>

The dashboard receives encrypted medication records from a simulated IoMT infusion device, verifies data integrity, performs anomaly detection using a Random Forest model, and generates anomaly or security alerts together with human-readable explanations.

---

## Dashboard

### Dashboard Homepage

![Dashboard Homepage](screenshots/Dashboard%20homepage.png)

### Live Stream Simulation

![Live Stream Simulation](screenshots/Live%20Stream%20Simulation%20Section.png)

### Anomaly Detection

![Anomaly Detection](screenshots/Anomaly%20detected.png)

### Explainable AI (XAI)

![XAI Explanation](screenshots/XAI%20explanation.png)

### Security Monitoring

![Security Monitoring](screenshots/Security%20log_HMAC%20verification.png)

---

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Cryptography
* Machine Learning
* Explainable AI (XAI)

---

## Dataset

This project uses data derived from the MIMIC-IV clinical database.

The original dataset is not included in this repository due to data usage restrictions.

---

## Author

**Nikoletta Andreou**

Bachelor Thesis Project, 2026

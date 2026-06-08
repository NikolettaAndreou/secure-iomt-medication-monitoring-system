# Secure IoMT Medication Monitoring System

**Bachelor Thesis Project – Applied Computer Science**

## Overview

This project presents a secure Internet of Medical Things (IoMT) medication monitoring system that combines machine learning, cybersecurity mechanisms, and explainable artificial intelligence (XAI) to detect anomalous medication infusion events.

The system was developed using clinical medication administration records from the MIMIC-IV database and evaluates multiple anomaly detection approaches under simulated cyberattack scenarios.

## System Architecture

![System Architecture](diagrams/Architecture%20diagram.png)

The architecture consists of six layers:

* Data Layer
* IoMT Simulation Layer
* Cybersecurity Layer
* AI Detection Layer
* Explainability Layer
* Visualization & Dashboard Layer

## Key Features

* Medication infusion monitoring
* Machine learning-based anomaly detection
* Explainable AI (XAI) support
* AES encryption for data confidentiality
* HMAC-SHA256 integrity verification
* Replay attack detection
* Interactive Streamlit dashboard
* Security event logging

## Threat Model

![Threat Model](diagrams/Threat%20Model%20Diagram.png)

The system evaluates and mitigates multiple healthcare IoMT threats including:

* Rate Manipulation
* Time Gap Manipulation
* Replay Attacks
* Data Tampering
* Spoofing Attempts

## Simulated Attack Scenarios

Four cyberattack scenarios were implemented and injected into the dataset:

1. Rate Manipulation
2. Sudden Rate Change
3. Time Gap Manipulation
4. Replay Attack

## Machine Learning Models

The following machine learning models were evaluated:

* Random Forest
* Logistic Regression
* Isolation Forest
* One-Class SVM
* Autoencoder

## Results

| Metric              | Random Forest |
| ------------------- | ------------- |
| Accuracy            | 95.75%        |
| Recall              | 97.9%         |
| False Negative Rate | 2.1%          |

The Random Forest model achieved the best overall performance and was selected as the final anomaly detection model.

### Model Performance Comparison

![Accuracy per Model](results/accuracy%20per%20model.png)

![False Positive Rate and False Negative Rate per Model](results/fpr%20and%20fnr%20per%20model.png)

![Recall and F1 Score per Model](results/recall%20and%20f1%20score.png)

### Attack Type Analysis

![Recall and F1 Score by Attack Type](results/results_by_attack_type_recall_f1.png)

## Dashboard Workflow

![Dashboard Workflow](diagrams/dashboard%20workflow-diagram.png)

The dashboard receives encrypted medication records from a simulated IoMT infusion device, verifies data integrity using HMAC-SHA256, extracts features, performs anomaly detection using the Random Forest model, and generates alerts, explanations, and security notifications.

## Dashboard

The Streamlit dashboard provides:

* Continuous data stream simulation
* Anomaly alerts
* Attack identification
* Security logs
* Rule-based explanations
* Feature importance visualizations

### Dashboard Homepage

![Dashboard Homepage](screenshots/Dashboard%20homepage.png)

### Live Stream Simulation

![Live Stream Simulation](screenshots/Live%20Stream%20Simulation%20Section.png)

### Anomaly Detection Example

![Anomaly Detected](screenshots/Anomaly%20detected.png)

### Explainable AI (XAI)

![XAI Explanation](screenshots/XAI%20explanation.png)

### Security Monitoring

![Security Log and HMAC Verification](screenshots/Security%20log_HMAC%20verification.png)

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Cryptography
* Machine Learning
* Explainable AI (XAI)

## Dataset

This project uses data derived from the MIMIC-IV clinical database.

The original dataset is not included in this repository due to data usage restrictions.

## Author

**Nikoletta Andreou**

Bachelor Thesis Project, 2026

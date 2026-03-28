# AI Based Real-Time Anomaly Detection and Autonomous Decision Support System for Deep Space Missions

An end-to-end machine learning system that monitors spacecraft telemetry data, detects anomalies, predicts component failures, and recommends autonomous corrective actions — simulating onboard AI for deep space missions where communication delays make real-time human intervention impractical.

## Problem Statement

Deep space missions (Chandrayaan-3, Aditya-L1, Mars Perseverance) face communication delays of 1.3 seconds to 24+ minutes. Spacecraft need onboard AI that can:
- **Detect anomalies** in sensor data autonomously
- **Predict failures** before they happen
- **Recommend actions** without waiting for ground control

This project simulates that system using the NASA C-MAPSS turbofan engine degradation dataset as a proxy for spacecraft subsystem telemetry.

## Results Summary

| Component | Model | Key Metric |
|-----------|-------|------------|
| Anomaly Detection | Isolation Forest | **ROC-AUC: 0.959** |
| Anomaly Detection | One-Class SVM | ROC-AUC: 0.954 |
| Anomaly Detection | Autoencoder (Keras) | ROC-AUC: 0.879 |
| RUL Prediction | LSTM (2-layer) | **RMSE: 15.25, R²: 0.867** |
| Decision System | Risk Engine (combined) | **Accuracy: 96%, Precision: 0.97** |

## Project Architecture

```
Sensor Data (21 channels)
        │
        ▼
┌─────────────────┐
│  Preprocessing   │  Remove low-variance sensors, normalize, compute RUL
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Anomaly │ │  LSTM  │
│Detect. │ │  RUL   │
│(3 models)│ │Predict.│
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌─────────────────┐
│  Risk Scoring    │  weighted: 40% anomaly + 60% RUL
│  Decision Engine │
└────────┬────────┘
         │
    ┌────┼────────────┐
    ▼    ▼            ▼
NOMINAL  CAUTION/    CRITICAL
         WARNING     (shut down)
```

## Dataset

**NASA C-MAPSS (FD001)** — Turbofan engine degradation simulation
- 100 engines run to failure
- 21 sensor channels + 3 operational settings
- ~20,631 data points

Download from: [Kaggle - NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

Place the files in the `data/` directory:
```
data/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

## Setup and Installation

```bash
# Create conda environment
conda create -n spacecraft python=3.10 -y
conda activate spacecraft

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Project Structure

```
spacecraft-anomaly-detection/
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   ├── train_FD001_clean.csv       # Preprocessed dataset
│   ├── anomaly_results.csv         # Anomaly scores from all models
│   ├── rul_predictions.csv         # LSTM RUL predictions
│   └── decision_log.csv            # Full autonomous decision log
├── notebooks/
│   ├── day1_eda.py                 # Data loading, EDA, preprocessing
│   ├── day2_anomaly_detection.py   # Isolation Forest, One-Class SVM, Autoencoder
│   ├── day3_lstm_rul.py            # LSTM for RUL prediction
│   └── day4_decision_engine.py     # Risk scoring + decision logic
├── results/
│   ├── sensor_trends_engine1.png
│   ├── engine_lifetimes.png
│   ├── correlation_heatmap.png
│   ├── anomaly_scores_engine1.png
│   ├── confusion_matrices.png
│   ├── roc_comparison.png
│   ├── lstm_training_history.png
│   ├── predicted_vs_actual_rul.png
│   ├── rul_predictions_engines.png
│   ├── risk_score_timeline.png
│   ├── decision_distribution.png
│   ├── risk_vs_rul.png
│   └── precision_recall_curve.png
└── report/
    └── technical_report.docx
```

## How to Run

Run the scripts in order:

```bash
python notebooks/day1_eda.py                 # Preprocess data
python notebooks/day2_anomaly_detection.py   # Train anomaly models
python notebooks/day3_lstm_rul.py            # Train LSTM
python notebooks/day4_decision_engine.py     # Run decision engine
```

## ML/AI Skills Demonstrated

- **Data preprocessing**: pandas, NumPy, MinMaxScaler, feature engineering
- **Unsupervised learning**: Isolation Forest, One-Class SVM
- **Deep learning**: Autoencoder (Keras), LSTM (TensorFlow)
- **Time-series analysis**: sliding window sequences, temporal feature extraction
- **Model evaluation**: ROC-AUC, precision-recall, confusion matrices, RMSE, R²
- **System integration**: multi-model fusion, risk scoring, autonomous decision logic

## Key Findings

1. **Simpler models can outperform deep learning** on structured tabular data — Isolation Forest (AUC 0.959) beat the Autoencoder (AUC 0.879) due to limited data size and low feature dimensionality.
2. **Model fusion improves performance** — combining anomaly detection with RUL prediction achieved 96% accuracy, exceeding any individual model.
3. **LSTM captures degradation trends** — the model predicts remaining life within ~12 cycles on average, providing sufficient lead time for corrective action.

## Submitted To

India Space Academy, Department of Space Education (ISW), New Delhi

## License

MIT License

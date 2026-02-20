# Time-Series Forecasting with Multi-Model Pipeline

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.101-green)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.4.1-orange)](https://mlflow.org/)

---

## 🔹 Overview

This repository provides a **full-stack time-series forecasting pipeline** supporting multiple models:

* **SARIMA** (classical statistical model)
* **Prophet** (additive model for time series forecasting)
* **XGBoost** (lag-based regression approach)

Features:

* Automated model training, evaluation, and logging with **MLflow**
* Dynamic **FastAPI endpoint** for predictions
* Safe **multi-model selection** and fallback to the best model
* Caching and utilities for production-ready deployments

---

## 📂 Repository Structure

```
Time-Series-Forecasting/
├── data/                  # Raw and processed datasets
├── src/
│   ├── api/
│   │   ├── app.py         # FastAPI application
│   │   └── schemas.py     # Pydantic request/response models
│   ├── pipelines/
│   │   ├── train.py       # Training pipeline for SARIMA, Prophet, XGBoost
│   │   └── evaluate.py    # RMSE, MAE, MAPE functions
│   ├── models/
│   │   └── sarima_model.py # SARIMA model training
│   └── utils/
│       ├── outliers.py    # Outlier detection & cleaning
│       ├── cache.py       # Prediction caching
│       └── model_handler.py # MLflow model loading utilities
├── Dockerfile
├── mlflow.db              # SQLite MLflow DB (optional, add to .gitignore)
├── requirements.txt
└── README.md
```

---

## ⚡ Features

1. **Multi-Model Forecasting**

   * Compare SARIMA, Prophet, and XGBoost on the same dataset
   * Automatically log metrics and plots to MLflow

2. **Dynamic Model Serving**

   * FastAPI `/predict` endpoint:

```json
{
  "model_name": "XGBoost",
  "values": [101.2, 102.5, 103.1, 102.8, 103.5, 104, 104.2]
}
```

* Returns forecasted values and model metadata

3. **MLflow Integration**

   * Track experiments, metrics, and artifacts
   * Automatically register the **best model** and allow fallback

4. **Production Safety**

   * Validates input length for lag-based models
   * Caching repeated requests for faster response

---

## 🛠 Installation

```bash
# Clone repo
git clone https://github.com/y123shiva/Time-Series-Forecasting.git
cd Time-Series-Forecasting

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Optional: Install Git LFS if your repo uses large files:

```bash
git lfs install
```

---

## 🚀 Running the Training Pipeline

```bash
python src/pipelines/train.py
```

* Trains SARIMA, Prophet, and XGBoost models
* Logs metrics and plots to MLflow
* Registers the **best model** as `financial_forecast_best`

---

## 🖥 Running the API

```bash
# Start FastAPI server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

* API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Example POST request:

```json
{
  "model_name": "XGBoost",
  "values": [101.2, 102.5, 103.1, 102.8, 103.5, 104, 104.2]
}
```

* Example response:

```json
{
  "predictions": [106.07, 106.49, 109.03, ...],
  "model_name": "XGBoost"
}
```

---

## 📈 MLflow UI

Start MLflow UI:

```bash
mlflow ui --host 0.0.0.0 --port 5001
```

* Dashboard: [http://127.0.0.1:5001](http://127.0.0.1:5001)
* View experiments, registered models, and plots

---

## 📝 Notes

* **Do not commit `mlflow.db`** — add it to `.gitignore`
* Minimum 7 historical values are required for XGBoost predictions
* Prophet and SARIMA models can handle full historical series dynamically

---

## 💡 Future Enhancements

* Auto-scaling FastAPI with Docker/Kubernetes
* Include **confidence intervals** for forecasts
* Web dashboard with **visual forecasts** and model comparison

---

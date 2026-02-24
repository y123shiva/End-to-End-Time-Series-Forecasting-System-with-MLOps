# 📈 Time Series Forecasting & MLOps System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2)
![Airflow](https://img.shields.io/badge/Orchestration-Airflow-017CEE)
![Status](https://img.shields.io/badge/Production-Ready-success)

A **production-grade, end-to-end forecasting and MLOps platform** that trains, evaluates, tracks, and deploys multiple time-series models with an interactive dashboard and scalable APIs.

---

# 🚀 Project Overview

This system predicts future demand (sales/bookings/traffic) using:

* Statistical models
* Machine learning models
* Automated pipelines
* Experiment tracking
* Production APIs
* Containerized deployment

It mirrors how forecasting systems are built at companies like:

* Amazon
* Uber
* Swiggy
* Walmart
* Flipkart

The focus is on:

✔ Reproducibility
✔ Modularity
✔ Scalability
✔ Production readiness
✔ Clean architecture

---

# 🎯 Business Problem

Organizations need accurate demand forecasts to:

* Prevent stock-outs
* Reduce overstock
* Optimize pricing
* Improve supply chain planning
* Make data-driven decisions

This system delivers **multi-model forecasting + automated evaluation** to select the best model dynamically.

---

# ✨ Key Highlights

✅ Multi-model forecasting (SARIMA, Prophet, XGBoost)
✅ Automated training pipelines
✅ Experiment tracking
✅ Interactive dashboard
✅ REST inference API
✅ Workflow orchestration
✅ Dockerized deployment
✅ Reproducible experiments
✅ Production-ready structure

---

# 🧠 Models Implemented

| Model              | Type        | Best For            |
| ------------------ | ----------- | ------------------- |
| statsmodels SARIMA | Statistical | Seasonality         |
| Prophet            | Additive    | Trend + holidays    |
| XGBoost            | ML          | Non-linear patterns |

---

# 🏗️ System Architecture

```
Data → Feature Engineering → Model Training → MLflow Tracking
        ↓
   Model Evaluation
        ↓
Best Model Selected
        ↓
FastAPI Inference API
        ↓
Streamlit Dashboard
        ↓
Docker Deployment
```

### Tech Stack

| Layer            | Tool                     |
| ---------------- | ------------------------ |
| API              | FastAPI                  |
| Dashboard        | Streamlit                |
| Tracking         | MLflow                   |
| Orchestration    | Apache Airflow           |
| Containerization | Docker                   |
| Models           | XGBoost, Prophet, SARIMA |
| Language         | Python                   |

---

# 📊 Dashboard Preview

### 🎥 Interactive Forecast Dashboard

👉 Streamlit URL -->> https://orange-acorn-wjp6vpqpgx5hgx4j-8501.app.github.dev/#financial-price-time-series-dashboard

Features:

* Model comparison
* RMSE/MAE/MAPE
* Historical vs forecast
* Run selection
* Experiment tracking

---

# 📈 Model Metrics Comparison

| Model   | RMSE     | MAE     | MAPE     |
| ------- | -------- | ------- | -------- |
| SARIMA  | 18.3     | 12.4    | 7.9%     |
| Prophet | 16.8     | 11.1    | 6.8%     |
| XGBoost | **14.2** | **9.5** | **5.2%** |

✔ Automatically selects best-performing model

---

# 📂 Project Structure

```
.
├── src/
│   ├── pipelines/
│   ├── models/
│   ├── utils/
│   └── evaluate.py
│
├── dashboard/          # Streamlit UI
├── dags/               # Airflow workflows
├── data/
├── notebooks/
├── mlruns/             # MLflow experiments
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# ⚙️ Setup Instructions

## 1️⃣ Clone

```bash
git clone <repo-url>
cd Time-Series-Forecasting
```

## 2️⃣ Install

```bash
pip install -r requirements.txt
```

## 3️⃣ Run Training

```bash
python src/train.py
```

## 4️⃣ Start API

```bash
uvicorn src.api.main:app --reload
```

Visit:

```
http://localhost:8000/docs
```

You can also trigger a full retraining via the API by POSTing to `/train`.
The Swagger UI shows the new endpoint once the server is running.

---

## 5️⃣ Launch Dashboard

```bash
cd dashboard
streamlit run app.py
```

---

## 6️⃣ Run with Docker

```bash
docker-compose up --build
```

---

# 🔄 MLflow Tracking

Start server:

```bash
mlflow ui
```

Open:

```
http://localhost:5000
```

Track:

* Parameters
* Metrics
* Artifacts
* Models
* Experiments

---

# 🚀 Deployment

## Streamlit Cloud

1. Push repo to GitHub
2. Deploy on Streamlit Cloud
3. Select `dashboard/app.py`

---

## Docker

```
docker build -t forecasting-app .
docker run -p 8000:8000 forecasting-app
```

---

# 🧪 Testing

```bash
pytest
```

Includes:

* Pipeline tests
* Model tests
* API tests

---

*** This project demonstrates:

✔ End-to-end ML ownership
✔ Production APIs
✔ Experiment tracking
✔ Workflow automation
✔ Containerization
✔ Clean architecture
✔ Real-world scalability

**Skills validated:**

* MLOps
* Time Series Forecasting
* Backend APIs
* System design
* Deployment engineering

Ideal for:

* Machine Learning Engineer
* Data Scientist
* MLOps Engineer
* Applied Scientist

---

# 📌 Future Improvements

* CI/CD pipeline
* Drift detection
* Auto-retraining
* Cloud deployment (AWS/GCP)
* Feature store

---

# 👩‍💻 Author

**Shivani Yadav**

Machine Learning Engineer | Time Series | MLOps | APIs

* GitHub: 
* LinkedIn: your-profile

---

# ⭐ If you found this helpful

Star ⭐ the repo to support the project!



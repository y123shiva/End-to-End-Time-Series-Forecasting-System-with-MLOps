Here’s your cleaned, final **README.md** content, ready to copy & paste as-is. It integrates your multi-model forecasting pipeline, FastAPI, MLflow, and Airflow setup:

---

# 📈 Financial Time Series Forecasting

This project provides a comprehensive framework for forecasting financial time series data using multiple machine learning models. It includes exploratory data analysis, ML model training, API serving, experiment tracking with MLflow, and orchestration with Apache Airflow.

---

## 🛠️ Technologies & Stack

* **Language:** Python 3
* **ML Models:** SARIMA, Prophet, XGBoost
* **Experiment Tracking:** MLflow
* **Orchestration:** Apache Airflow
* **API:** FastAPI
* **Containerization:** Docker & Docker Compose
* **Key Libraries:**

  * Pandas, NumPy, Scikit-learn
  * Statsmodels (SARIMA), Prophet
  * XGBoost
  * Matplotlib, Seaborn
  * MLflow

---

## 📁 Project Structure

```
├── src/                          
│   ├── api/                      
│   │   ├── app.py               
│   │   └── schemas.py           
│   ├── models/                   
│   │   ├── sarima_model.py      
│   │   ├── prophet_model.py     
│   │   └── xgb_model.py         
│   ├── pipelines/                
│   │   ├── train.py             
│   │   ├── data_loader.py       
│   │   └── evaluate.py          
│   ├── utils/                    
│   │   ├── model_handler.py     
│   │   ├── features.py          
│   │   ├── scaler.py            
│   │   ├── cache.py             
│   │   └── outliers.py          
│   └── config.py                
├── dags/                         
│   └── forecast_dag.py          
├── notebooks/                    
│   └── exploration.ipynb        
├── tests/                        
├── data/                         
│   └── financial_data.csv       
├── mlruns/                       
├── airflow/                      
├── Dockerfile                    
├── docker-compose.yml            
├── requirements.txt             
└── README.md                    
```

---

## 🔍 Key Features

### 1. Exploratory Data Analysis (EDA)

* Dataset structure, missing values, outliers visualization
* Stationarity tests (ADF), ACF/PACF plots
* Seasonal decomposition and trend analysis

### 2. Data Preprocessing

* Log transformation for variance stabilization
* Outlier detection and smoothing
* Train/test split (80%/20%)
* Scaling and feature engineering for ML models

### 3. Multiple Forecasting Models

* **SARIMA:** Seasonal ARIMA for stationary/seasonal data
* **Prophet:** Handles trends, seasonality, holidays, robust to outliers
* **XGBoost:** Gradient boosting for complex, non-linear patterns

### 4. Model Evaluation

* RMSE, MAE, MAPE
* Rolling window cross-validation
* Multi-model comparison

### 5. REST API

* Serve trained models via FastAPI
* Prediction endpoints with model selection
* Request/response validation using Pydantic schemas

### 6. Experiment Tracking

* MLflow for logging models, metrics, and artifacts
* Model registry and versioning

### 7. Workflow Orchestration

* Airflow DAGs for automated retraining and forecasting
* Scheduled jobs with error handling and retries

---

## 🚀 Getting Started

### Prerequisites

* Docker & Docker Compose
* Python 3.8+

### Option 1: Docker Compose

```bash
git clone https://github.com/y123shiva/Time-Series-Forecasting.git
cd Time-Series-Forecasting

docker-compose up -d
```

* API: [http://localhost:5000](http://localhost:5000)
* Airflow: [http://localhost:8080](http://localhost:8080)
* MLflow: [http://localhost:5000](http://localhost:5000)

### Option 2: Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Run Training Pipeline

```bash
python src/pipelines/train.py
```

### Start API Server

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 5000
```

### Start MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### Start Airflow

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow webserver --port 8080
airflow scheduler
```

---

## 📊 Model Comparison

| Model   | Approach            | Strengths                                              | Best For                                   |
| ------- | ------------------- | ------------------------------------------------------ | ------------------------------------------ |
| SARIMA  | Statistical         | Interpretable, seasonal patterns, confidence intervals | Stable seasonal data, economic indicators  |
| Prophet | Trend + Seasonality | Robust to outliers, handles holidays, simple tuning    | Business metrics, web traffic              |
| XGBoost | ML-based            | Non-linear patterns, high accuracy, feature importance | Complex relationships, high-frequency data |

---

## 📈 Sample API Usage

```bash
# SARIMA forecast
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "SARIMA",
    "values": [101.2, 102.5, 103.1, 102.8, 103.5, 104, 104.2]
  }'
```

```json
{
  "predictions": [106.07, 106.49, 109.03, ...],
  "model_name": "SARIMA"
}
```

---

## ⚙️ Configuration

Edit `src/config.py` to adjust:

* Model hyperparameters (SARIMA, Prophet, XGBoost)
* Data paths
* API settings
* MLflow URI and registry
* Airflow DAG intervals

---

## 🧪 Testing

```bash
pytest tests/ --cov=src
```

---

## 📝 Contributing

1. Fork repo → `git checkout -b feature/your-feature`
2. Commit changes → `git commit -am "Add feature"`
3. Push branch → `git push origin feature/your-feature`
4. Submit a pull request

---

## 📄 License

MIT License. See LICENSE file for details.

---

## 👨‍💻 Author

**y123shiva** – For questions, open an issue on GitHub.

---

This version is clean, complete, and ready for your repo.

If you want, I can also create a **short “Quick Start” section with Docker + API + MLflow launch in one command** to make it ultra user-friendly.

Do you want me to do that?

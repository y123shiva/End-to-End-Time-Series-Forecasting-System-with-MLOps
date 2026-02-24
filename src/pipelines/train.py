import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.prophet
import matplotlib.pyplot as plt
import time
from typing import Optional

from src.pipelines.data_loader import load_data
from src.utils.outliers import detect, clean
from src.models.sarima_model import train_sarima
from src.pipelines.evaluate import rmse, mae, mape
from src.config import (
    OUTLIER_CONFIG, TRAIN_TEST_SPLIT, SARIMA_CONFIG,
    PROPHET_CONFIG, XGBOOST_CONFIG, MODEL_REGISTRY_STAGES
)

# prophet is imported lazily within the training function below to
# avoid heavy dependencies during light-weight imports (e.g. tests). 
import xgboost as xgb

# -------------------------
# Feature Engineering
# -------------------------

def create_lags(series, n_lags=7):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# -------------------------
# Model Training Functions
# -------------------------

def train_prophet(df, test_size):
    # lazy import so the module can be imported in lightweight contexts
    from prophet import Prophet

    df = df.copy().reset_index()
    df.columns = ["ds", "y"]
    df['ds'] = pd.to_datetime(df['ds'])

    train = df[:-test_size]
    test = df[-test_size:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=test_size)
    forecast = model.predict(future)
    preds = forecast["yhat"].tail(test_size).values

    return model, preds, test["y"].values

def train_xgb(series, test_size, n_lags=None):
    if n_lags is None:
        n_lags = XGBOOST_CONFIG["n_lags"]
    
    X, y = create_lags(series, n_lags=n_lags)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = xgb.XGBRegressor(
        n_estimators=XGBOOST_CONFIG["n_estimators"],
        learning_rate=XGBOOST_CONFIG["learning_rate"],
        max_depth=XGBOOST_CONFIG["max_depth"],
        subsample=XGBOOST_CONFIG["subsample"],
        colsample_bytree=XGBOOST_CONFIG["colsample_bytree"],
        random_state=XGBOOST_CONFIG["random_state"]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return model, preds, y_test

# -------------------------
# MLflow Logging
# -------------------------

def log_model_run(name, model, y_true, y_pred):
    r = rmse(y_true, y_pred)
    m = mae(y_true, y_pred)
    p = mape(y_true, y_pred)

    metrics = {"rmse": r, "mae": m, "mape": p}

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_param("model", name)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Plot forecast
        timestamp = int(time.time())
        plot_path = f"{name}_forecast_{timestamp}.png"
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        plt.title(name)
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        # Log model according to type
        try:
            if name == "Prophet":
                mlflow.prophet.log_model(model, artifact_path="model", registered_model_name=name)
            else:
                mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=name)
        except Exception as e:
            print(f"⚠ Could not log model {name}: {e}")

    print(f"{name} → RMSE:{r:.3f} MAE:{m:.3f} MAPE:{p:.2f}%")
    return metrics

# -------------------------
# Safe Model Loader for FastAPI
# -------------------------

def load_model_safe(model_name: str):
    try:
        return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    except Exception:
        print(f"⚠ Model {model_name} not found in Production, using fallback")
        return mlflow.pyfunc.load_model("models:/financial_forecast_best/Production")

# -------------------------
# MAIN PIPELINE
# -------------------------

def run(model_name: Optional[str] = None):
    """Main training pipeline.

    If **model_name** is supplied it is currently ignored and the full multi-model
    comparison is executed; the parameter exists for future extension.

    Returns a summary dictionary containing the per-model scores and the name of
    the best model so that callers (e.g. a web API) can present results programmatically.
    """
    mlflow.set_experiment("financial_forecast_multi_model")
    df = load_data("data/financial_data.csv")
    series = df["Price"]

    # Outlier detection & cleaning
    mask = detect(series, **OUTLIER_CONFIG)
    series = clean(series, mask)

    # Train-test split
    split = int(len(series) * TRAIN_TEST_SPLIT)
    train, test = series[:split], series[split:]
    test_size = len(test)

    scores = {}
    models = {}

    with mlflow.start_run(run_name="Multi-Model-Comparison"):
        mlflow.log_params({
            "train_test_split": TRAIN_TEST_SPLIT,
            "test_size": test_size,
            "outlier_contamination": OUTLIER_CONFIG["contamination"]
        })

        # 1️⃣ SARIMA
        try:
            sarima_model = train_sarima(train, **SARIMA_CONFIG)
            sarima_preds = sarima_model.forecast(test_size)
            scores["SARIMA"] = log_model_run("SARIMA", sarima_model, test, sarima_preds)
            models["SARIMA"] = sarima_model
            print("✅ SARIMA training completed")
        except Exception as e:
            print(f"⚠ SARIMA training failed: {e}")
            scores["SARIMA"] = {"rmse": float('inf'), "mae": float('inf'), "mape": float('inf')}

        # 2️⃣ Prophet
        try:
            prophet_model, prophet_preds, prophet_test = train_prophet(series.to_frame(), test_size)
            scores["Prophet"] = log_model_run("Prophet", prophet_model, prophet_test, prophet_preds)
            models["Prophet"] = prophet_model
            print("✅ Prophet training completed")
        except Exception as e:
            print(f"⚠ Prophet training failed: {e}")
            scores["Prophet"] = {"rmse": float('inf'), "mae": float('inf'), "mape": float('inf')}

        # 3️⃣ XGBoost
        try:
            xgb_model, xgb_preds, xgb_test = train_xgb(series.values, test_size)
            scores["XGBoost"] = log_model_run("XGBoost", xgb_model, xgb_test, xgb_preds)
            models["XGBoost"] = xgb_model
            print("✅ XGBoost training completed")
        except Exception as e:
            print(f"⚠ XGBoost training failed: {e}")
            scores["XGBoost"] = {"rmse": float('inf'), "mae": float('inf'), "mape": float('inf')}

        # -------------------------
        # Register all models + set best to Production
        # -------------------------
        best_model_name = min(scores, key=lambda k: scores[k]["rmse"])
        best_model_object = models[best_model_name]
        best_score = scores[best_model_name]

        mlflow.set_tag("best_model", best_model_name)
        mlflow.log_metric("best_rmse", best_score["rmse"])

        # Register each model with appropriate stage
        for model_name in ["SARIMA", "Prophet", "XGBoost"]:
            if model_name not in models:
                print(f"⏭ Skipping registration for {model_name} (failed training)")
                continue

            try:
                model_obj = models[model_name]
                stage = "Production" if model_name == best_model_name else "Staging"
                
                if model_name == "Prophet":
                    mlflow.prophet.log_model(
                        model_obj,
                        artifact_path=model_name.lower(),
                        registered_model_name=model_name
                    )
                else:
                    mlflow.sklearn.log_model(
                        model_obj,
                        artifact_path=model_name.lower(),
                        registered_model_name=model_name
                    )
                print(f"✅ Registered {model_name} model (stage: {stage})")
                
                # Move to appropriate stage in registry
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                latest_version = client.get_latest_versions(model_name, stages=None)[0]
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage=stage,
                    archive_existing_versions=False
                )
                print(f"✅ Moved {model_name} to {stage} stage")
                
            except Exception as e:
                print(f"⚠ Could not register {model_name}: {e}")

        # Also register a generic "financial_forecast_best" for backward compatibility
        try:
            if best_model_name == "Prophet":
                mlflow.prophet.log_model(
                    best_model_object,
                    artifact_path="best_model",
                    registered_model_name="financial_forecast_best"
                )
            else:
                mlflow.sklearn.log_model(
                    best_model_object,
                    artifact_path="best_model",
                    registered_model_name="financial_forecast_best"
                )
            print(f"✅ Registered best model: {best_model_name} as financial_forecast_best")
        except Exception as e:
            print(f"⚠ Could not register best model: {e}")

    print("\n" + "="*50)
    print("Model Comparison Results:")
    print("="*50)
    for k, v in scores.items():
        if v["rmse"] != float('inf'):
            print(f"{k:12} → RMSE: {v['rmse']:.4f}, MAE: {v['mae']:.4f}, MAPE: {v['mape']:.2f}%")
        else:
            print(f"{k:12} → FAILED")
    print(f"\n🏆 Best Model: {best_model_name}")
    print("="*50)

    # sanitize scores: JSON cannot represent infinite values so convert
    # to None before returning to any callers (e.g. API response)
    def _sanitize(val):
        import math
        if isinstance(val, float) and not math.isfinite(val):
            return None
        return val

    clean_scores = {}
    for m, met in scores.items():
        clean_scores[m] = {k: _sanitize(v) for k, v in met.items()}

    # return summary for callers
    return {"scores": clean_scores, "best_model": best_model_name}

if __name__ == "__main__":
    run()

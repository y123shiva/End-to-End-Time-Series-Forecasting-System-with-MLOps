import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from src.pipelines.data_loader import load_data
from src.utils.outliers import detect, clean
from src.models.sarima_model import train_sarima
from src.pipelines.evaluate import rmse, mae, mape

from prophet import Prophet
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
# Models
# -------------------------

def train_prophet(df, test_size):

    df = df.copy().reset_index()
    df.columns = ["ds", "y"]

    train = df[:-test_size]
    test = df[-test_size:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=test_size)
    forecast = model.predict(future)

    preds = forecast["yhat"].tail(test_size).values

    return model, preds, test["y"].values


def train_xgb(series, test_size):

    X, y = create_lags(series)

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return model, preds, y_test


# -------------------------
# MLflow logging
# -------------------------

def log_model_run(name, model, y_true, y_pred):

    r = rmse(y_true, y_pred)
    m = mae(y_true, y_pred)
    p = mape(y_true, y_pred)

    with mlflow.start_run(run_name=name, nested=True):

        mlflow.log_param("model", name)
        mlflow.log_metric("rmse", r)
        mlflow.log_metric("mae", m)
        mlflow.log_metric("mape", p)

        # forecast plot
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        plt.title(name)

        plot_path = f"{name}_forecast.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)

        # save model
        try:
            mlflow.sklearn.log_model(model, name="model")
        except:
            pass

    print(f"{name} → RMSE:{r:.3f} MAE:{m:.3f} MAPE:{p:.2f}%")

    return r


# -------------------------
# MAIN PIPELINE
# -------------------------

def run():

    mlflow.set_experiment("financial_forecast_multi_model")

    df = load_data("data/financial_data.csv")
    series = df["Price"]

    # outlier cleaning
    mask = detect(series)
    series = clean(series, mask)

    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]
    test_size = len(test)

    scores = {}

    # Parent run for grouping
    with mlflow.start_run(run_name="Multi-Model-Comparison"):

        # 1️⃣ SARIMA
        sarima_model = train_sarima(train)
        sarima_preds = sarima_model.forecast(test_size)

        scores["SARIMA"] = log_model_run(
            "SARIMA", sarima_model, test, sarima_preds
        )

        # 2️⃣ Prophet
        prophet_model, prophet_preds, prophet_test = train_prophet(
            series.to_frame(), test_size
        )

        scores["Prophet"] = log_model_run(
            "Prophet", prophet_model, prophet_test, prophet_preds
        )

        # 3️⃣ XGBoost
        xgb_model, xgb_preds, xgb_test = train_xgb(
            series.values, test_size
        )

        scores["XGBoost"] = log_model_run(
            "XGBoost", xgb_model, xgb_test, xgb_preds
        )

        # -------------------------
        # Comparison
        # -------------------------
        best_model = min(scores, key=scores.get)

        mlflow.set_tag("best_model", best_model)
        mlflow.log_metric("best_rmse", scores[best_model])

    print("\nModel Comparison:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    print(f"\nBest Model → {best_model}")


if __name__ == "__main__":
    run()

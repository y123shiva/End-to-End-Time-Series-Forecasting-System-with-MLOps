import numpy as np
import pandas as pd


# -------------------------
# Core metrics
# -------------------------

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# -------------------------
# Aggregator
# -------------------------

def all_metrics(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred)
    }


def comparison_table(results_dict):
    """
    results_dict = {
        "SARIMA": (y_true, y_pred),
        "XGBoost": (y_true, y_pred)
    }
    """
    rows = []

    for name, (y, p) in results_dict.items():
        m = all_metrics(y, p)
        m["model"] = name
        rows.append(m)

    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("rmse")

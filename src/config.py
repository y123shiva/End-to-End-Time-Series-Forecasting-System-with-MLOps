"""
Model configuration and hyperparameters.
Centralized settings for model training.
"""

# -------------------------
# Outlier Detection
# -------------------------
OUTLIER_CONFIG = {
    "contamination": 0.05,  # Expected % of outliers
    "random_state": 42
}

# -------------------------
# Data Splitting
# -------------------------
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test

# -------------------------
# SARIMA Configuration
# -------------------------
SARIMA_CONFIG = {
    "order": (1, 1, 1),           # (p, d, q)
    "seasonal_order": (1, 1, 1, 7),  # (P, D, Q, s) with s=7 for weekly
    "enforce_stationarity": False,
    "enforce_invertibility": False
}

# -------------------------
# Prophet Configuration
# -------------------------
PROPHET_CONFIG = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": 0.95,
    "growth": "linear"
}

# -------------------------
# XGBoost Configuration
# -------------------------
XGBOOST_CONFIG = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_lags": 7  # Number of lag features
}

# -------------------------
# Model Registry Stages
# -------------------------
# After comparison:
# - Best model → Production stage (for API serving)
# - Other models → Staging stage (for testing/comparison)
MODEL_REGISTRY_STAGES = {
    "SARIMA": "Staging",
    "Prophet": "Staging",
    "XGBoost": "Staging"
}

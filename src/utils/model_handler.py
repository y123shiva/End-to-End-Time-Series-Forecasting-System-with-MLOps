"""
Universal model handler for different forecasting model types.
Abstracts SARIMA, Prophet, and XGBoost prediction logic.
"""
import numpy as np
import pandas as pd
from typing import Union, Tuple
import joblib
import pickle


class ModelHandler:
    """Wrapper for different model types with unified prediction interface."""
    
    def __init__(self, model, model_type: str, n_lags: int = 7):
        """
        Initialize handler for a specific model type.
        
        Parameters:
        -----------
        model : object
            The trained model (SARIMA, Prophet, or XGBoost)
        model_type : str
            One of ["SARIMA", "Prophet", "XGBoost"]
        n_lags : int
            Number of lags for XGBoost models
        """
        self.model = model
        self.model_type = model_type
        self.n_lags = n_lags
        
        if model_type not in ["SARIMA", "Prophet", "XGBoost"]:
            raise ValueError(f"model_type must be SARIMA, Prophet, or XGBoost. Got {model_type}")
    
    def _unwrap_model(self):
        """
        Extract underlying model from MLflow PyFuncModel wrapper.
        
        MLflow wraps models differently based on type:
        - SARIMA/XGBoost (sklearn): _model_impl.sklearn_model
        - Prophet: _model_impl.pr_model
        """
        # Check if it's a PyFuncModel wrapper
        if hasattr(self.model, '_model_impl'):
            impl = self.model._model_impl
            
            # Prophet Model
            if hasattr(impl, 'pr_model'):
                return impl.pr_model
            
            # SARIMA and XGBoost (both are sklearn)
            if hasattr(impl, 'sklearn_model'):
                return impl.sklearn_model
        
        # If already unwrapped, return as-is
        return self.model
    
    def forecast(self, history: list, horizon: int = 1) -> np.ndarray:
        """
        Forecast future values using the appropriate model logic.
        
        Parameters:
        -----------
        history : list
            Historical values to use as context
        horizon : int
            Number of steps to forecast
            
        Returns:
        --------
        np.ndarray
            Predicted values of length `horizon`
        """
        if len(history) < self.n_lags:
            raise ValueError(
                f"History length ({len(history)}) must be >= n_lags ({self.n_lags})"
            )
        
        if self.model_type == "SARIMA":
            return self._forecast_sarima(history, horizon)
        elif self.model_type == "Prophet":
            return self._forecast_prophet(history, horizon)
        elif self.model_type == "XGBoost":
            return self._forecast_xgboost(history, horizon)
    
    def _forecast_sarima(self, history: list, horizon: int) -> np.ndarray:
        """SARIMA uses built-in forecast method."""
        # Handle MLflow PyFuncModel wrapper
        model = self._unwrap_model()
        return model.forecast(steps=horizon)
    
    def _forecast_prophet(self, history: list, horizon: int) -> np.ndarray:
        """
        Prophet requires future dataframe and returns yhat column.
        """
        model = self._unwrap_model()
        
        # Create future dates
        last_date = pd.Timestamp.now()  
        future_dates = pd.date_range(start=last_date, periods=horizon, freq='D')
        future = pd.DataFrame({'ds': future_dates})
        
        forecast = model.predict(future)
        return forecast['yhat'].values
    
    def _forecast_xgboost(self, history: list, horizon: int) -> np.ndarray:
        """
        XGBoost iteratively predicts using lag features.
        Each prediction gets appended to history for next step.
        """
        model = self._unwrap_model()
        
        predictions = []
        # Use last n_lags values
        working_history = list(history[-self.n_lags:])
        
        for _ in range(horizon):
            # Create lag features from recent history
            X = np.array(working_history[-self.n_lags:]).reshape(1, -1)
            next_pred = model.predict(X)[0]
            predictions.append(next_pred)
            working_history.append(next_pred)
        
        return np.array(predictions)

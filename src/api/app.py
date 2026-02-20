import numpy as np
from fastapi import FastAPI, HTTPException, Query
from typing import Optional, Dict
import mlflow.pyfunc
from src.api.schemas import PredictRequest, PredictResponse
from src.utils.cache import lru_forecast_cache
from src.utils.model_handler import ModelHandler

N_LAGS = 7
AVAILABLE_MODELS = ["SARIMA", "Prophet", "XGBoost"]
MAX_FORECAST_HORIZON = 365

# -------------------------
# Model registry with type tracking
# -------------------------
model_registry: Dict[str, Optional[ModelHandler]] = {}

def load_model(model_name: str) -> Optional[ModelHandler]:
    """
    Load model from MLflow registry and wrap with appropriate handler.
    
    Tries to load from Production stage first, then Staging, then None.
    Detects model type and returns ModelHandler for unified prediction interface.
    """
    if model_name in model_registry:
        return model_registry[model_name]
    
    # Try different stages
    stages_to_try = ["Production", "Staging", None]
    
    for stage in stages_to_try:
        try:
            if stage:
                uri = f"models:/{model_name}/{stage}"
            else:
                # Try latest version
                uri = f"models:/{model_name}/latest"
            
            model = mlflow.pyfunc.load_model(uri)
            
            # Detect model type from underlying implementation
            model_type = _detect_model_type(model, model_name)
            
            # Wrap with handler for unified interface
            handler = ModelHandler(model, model_type, n_lags=N_LAGS)
            model_registry[model_name] = handler
            print(f"✅ Loaded {model_type} model: {model_name} (stage: {stage or 'latest'})")
            return handler
            
        except Exception as e:
            continue
    
    # All attempts failed
    print(f"⚠ Could not load Production model '{model_name}' from any stage")
    return None

def _detect_model_type(model, model_name: str) -> str:
    """
    Detect the underlying model type by checking model properties.
    """
    try:
        # Check MLflow model metadata
        if hasattr(model, 'metadata'):
            flavor = model.metadata.flavors if hasattr(model.metadata, 'flavors') else {}
            if 'prophet' in flavor:
                return "Prophet"
            elif 'statsmodels' in flavor:
                return "SARIMA"
        
        # Fallback: check model_name
        if "prophet" in model_name.lower():
            return "Prophet"
        elif "sarima" in model_name.lower():
            return "SARIMA"
        else:
            return "XGBoost"
    except:
        return "XGBoost"  # Default fallback

# -------------------------
# FastAPI initialization
# -------------------------
app = FastAPI(
    title="Financial Forecast API",
    description="Multi-model time-series forecasting service",
    version="1.0"
)

@app.get("/")
def health():
    """Health check endpoint."""
    return {"status": "ok", "available_models": AVAILABLE_MODELS}

# -------------------------
# Cached prediction function
# -------------------------
@lru_forecast_cache(maxsize=256)
def cached_forecast(values: tuple, model_name: str, horizon: int) -> list:
    """
    Generate predictions using appropriate model handler.
    Results cached by (values, model_name, horizon) tuple.
    """
    handler = load_model(model_name)
    if handler is None:
        raise HTTPException(status_code=500, detail=f"Model {model_name} not available")
    
    try:
        preds = handler.forecast(list(values), horizon=horizon)
        return preds.tolist()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# -------------------------
# /predict → next value
# -------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict next single value using specified model."""
    model_name = req.model_name or "XGBoost"
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Model must be one of {AVAILABLE_MODELS}"
        )
    
    if len(req.values) < N_LAGS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {N_LAGS} historical values"
        )
    
    preds = cached_forecast(tuple(req.values), model_name, horizon=1)
    return PredictResponse(predictions=preds, model_name=model_name)

# -------------------------
# /forecast → multi-step horizon
# -------------------------
@app.post("/forecast", response_model=PredictResponse)
def forecast(req: PredictRequest, horizon: int = Query(30, gt=0, le=MAX_FORECAST_HORIZON)):
    """Forecast multiple steps ahead using specified model."""
    model_name = req.model_name or "XGBoost"
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be one of {AVAILABLE_MODELS}"
        )
    
    if len(req.values) < N_LAGS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {N_LAGS} historical values"
        )
    
    preds = cached_forecast(tuple(req.values), model_name, horizon=horizon)
    return PredictResponse(predictions=preds, model_name=model_name)

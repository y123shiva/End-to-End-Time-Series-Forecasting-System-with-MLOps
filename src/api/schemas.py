from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PredictRequest(BaseModel):
    values: List[float]
    model_name: Optional[str] = "XGBoost"

class PredictResponse(BaseModel):
    predictions: List[float]
    model_name: str

# request/response models for training endpoint
default_train_message = "Triggered training pipeline (may take a few seconds)"

class TrainRequest(BaseModel):
    # optional model name to focus training (currently ignored, full pipeline runs)
    model_name: Optional[str] = None

class TrainResponse(BaseModel):
    success: bool
    message: str
    best_model: Optional[str] = None
    metrics: Optional[Dict[str, Dict[str, float]]] = None

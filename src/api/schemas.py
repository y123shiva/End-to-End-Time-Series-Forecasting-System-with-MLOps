from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    values: List[float]
    model_name: Optional[str] = "XGBoost"

class PredictResponse(BaseModel):
    predictions: List[float]
    model_name: str

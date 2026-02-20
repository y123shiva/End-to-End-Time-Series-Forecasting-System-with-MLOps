import numpy as np
import mlflow.pyfunc
from fastapi import FastAPI

app = FastAPI(title="Financial Forecast API")

MODEL_URI = "runs:/latest/model"  # latest best model

model = None


@app.on_event("startup")
def load_model():
    global model
    model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
def predict(values: list[float]):

    X = np.array(values).reshape(1, -1)
    pred = model.predict(X)

    return {"forecast": float(pred[0])}

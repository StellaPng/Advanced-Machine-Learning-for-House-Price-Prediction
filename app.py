from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Load the trained model
model = joblib.load("Models/tuned_xgb_model.pkl")

# 2. Feature names used for prediction (must match training order)
feature_names = [
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'TotalBsmtSF',
    'FullBath',
    'YearBuilt',
    'YearRemodAdd',
    'MasVnrArea',
    'Fireplaces',
    'OpenPorchSF_log'
]

# 3. Define the request body model using Pydantic
class HouseFeatures(BaseModel):
    OverallQual: float
    GrLivArea: float
    GarageCars: float
    TotalBsmtSF: float
    FullBath: float
    YearBuilt: float
    YearRemodAdd: float
    MasVnrArea: float
    Fireplaces: float
    OpenPorchSF_log: float

app = FastAPI()

@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Predict the house sale price given the input features.
    """
    # Extract features in the required order
    input_data = np.array([[getattr(features, f) for f in feature_names]])
    # Predict using the loaded model
    prediction = model.predict(input_data)[0]
    return {"predicted_sale_price": float(prediction)}

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("pricing_model.pkl")

app = FastAPI(title="Pricing API")

class PriceInput(BaseModel):
    cost_price: float
    demand: float
    inventory: int
    competitor_price: float

@app.get("/")
def home():
    return {"message": "Pricing API is running"}

@app.post("/predict")
def predict_price(data: PriceInput):
    X = np.array([[ 
        data.cost_price,
        data.demand,
        data.inventory,
        data.competitor_price
    ]])
    price = model.predict(X)[0]
    return {"recommended_price": round(float(price), 2)}
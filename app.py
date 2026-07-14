import sys
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from src.Pipeline.PredictionPipeline.PredictionPipepline import PredictionPipeline
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

# 1. Exact Column Mapping Schema
class TransactionInput(BaseModel):
    transaction_id: int
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age: int

class PredictionResponse(BaseModel):
    transaction_id: int
    is_fraud: int
    status: str

@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(data: TransactionInput):
    try:
        logging.info(f"Received transaction {data.transaction_id} for prediction.")
        
        # Convert incoming JSON to clean dictionary, then to Pandas DataFrame
        input_dict = [data.model_dump()]
        df_payload = pd.DataFrame(input_dict)
        
        # Run prediction pipeline
        predictor = PredictionPipeline()
        prediction_result = predictor.Predict(input_features=df_payload)
        
        final_prediction = int(prediction_result[0])
        status_message = "ALERT: Fraudulent!" if final_prediction == 1 else "Approved Safe"
        
        return PredictionResponse(
            transaction_id=data.transaction_id,
            is_fraud=final_prediction,
            status=status_message
        )
        
    except Exception as e:
        logging.error(f"Error handling prediction request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",          
        host="127.0.0.1",   
        port=8080,
        reload=True
    )
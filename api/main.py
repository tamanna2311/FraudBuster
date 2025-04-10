"""
File Name: main.py
Purpose: The FastAPI Web Server entry point.
Why it exists: Without this, our python code is just a local script. This exposes it as a REST API endpoint that a web/mobile app could actually call over HTTP.
Files that depend on this: None
Files that use this: None (Ran by uvicorn webserver)
Inputs: HTTP Requests
Outputs: HTTP Responses (JSON)
"""

from fastapi import FastAPI, HTTPException
from api.schemas import TransactionRequest, PredictionResponse
from src.predict import predict_fraud
import json

# Initialize FastAPI application
app = FastAPI(
    title="FraudBuster API",
    description="Near real-time credit card fraud detection system.",
    version="1.0.0"
)

@app.get("/")
@app.get("/health")
def health_check():
    """
    What it does: A simple ping endpoint.
    Why it is needed: In a real cloud environment (AWS/GCP), the infrastructure will constantly ping /health to see if the server is alive or crashed.
    """
    return {"status": "ok", "message": "FraudBuster API is running smoothly."}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: TransactionRequest):
    """
    What it does: Receives a JSON payload, passes it to the AI brain, and returns the result.
    Why it is needed: This is the actual functional service endpoint of the project.
    Inputs: The JSON validated by Pydantic (TransactionRequest)
    Outputs: The JSON response (PredictionResponse)
    """
    # Convert Pydantic object to simple dictionary
    data_dict = request.model_dump()
    
    # Send to the Machine Learning pipeline
    try:
        result = predict_fraud(data_dict)
        
        # If the predictor found an error (like missing models)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        # Return cleanly
        return PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_prediction=result["fraud_prediction"],
            fraud_probability=result["fraud_probability"],
            risk_factors=result["risk_factors"]
        )
        
    except Exception as e:
        # Catch unexpected crashes safely
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

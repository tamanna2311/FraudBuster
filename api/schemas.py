"""
File Name: schemas.py
Purpose: Defines the exact shape of data the API is allowed to receive and send.
Why it exists: If the API expects 'transaction_amount' as a number, but gets 'five dollars' as text, Pydantic immediately rejects it safely with a 422 Error instead of crashing our python model.
Files that depend on this: None
Files that use this: api/main.py
Inputs: None
Outputs: Pydantic classes.
"""

from pydantic import BaseModel, Field

class TransactionRequest(BaseModel):
    """
    What it does: Maps out the incoming JSON request.
    Why it is needed: Validation and security.
    """
    transaction_id: str = Field(..., description="Unique ID for the transaction")
    customer_id: str = Field(..., description="Customer who made the transaction")
    transaction_amount: float = Field(..., description="Cost of the transaction")
    transaction_time: str = Field(..., description="Time in ISO string format")
    merchant_category: str = Field(..., description="Type of merchant (e.g. electronics)")
    is_international: int = Field(..., description="0 for domestic, 1 for international")
    
    # These are features that an upstream system would have calculated right before hitting us
    time_since_last_txn: float = Field(0.0, description="Seconds since last purchase")
    user_txn_count: int = Field(0, description="Number of transactions recently")
    user_mean_spend: float = Field(0.0, description="Average spend history")
    dev_from_mean: float = Field(1.0, description="Ratio of current spend to mean spend")
    transaction_velocity_24h: int = Field(0, description="Velocity in 24 hours")

class PredictionResponse(BaseModel):
    """
    What it does: Maps out the JSON response we send back to the user/client.
    """
    transaction_id: str
    fraud_prediction: int
    fraud_probability: float
    risk_factors: str

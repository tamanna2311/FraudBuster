"""
File Name: predict.py
Purpose: Takes a single transaction and pushes it through the saved model to get a fraud prediction.
Why it exists: It bridges the gap between Web Server and Machine Learning model. The Web Server shouldn't know HOW to predict, it just asks predict.py to do it.
Files that depend on this: src/config.py, src/utils.py
Files that use this: api/main.py
Inputs: A dictionary containing transaction data.
Outputs: Fraud prediction (0 or 1), probability, and short explanation.
"""

import pandas as pd
from src.config import RF_MODEL_PATH, SCALER_PATH, TRAIN_X_PATH
from src.utils import load_object

# Load models into memory once when the file is imported
# This prevents the web server from reloading the hard drive file 100 times per second on high traffic!
print("[*] predictor engine initializing... loading models into RAM")
try:
    rf_model = load_object(RF_MODEL_PATH)
    scaler = load_object(SCALER_PATH)
    
    # We need to know the EXACT columns the model was trained on
    # So we peek at the training data columns
    training_cols = pd.read_csv(TRAIN_X_PATH, nrows=0).columns.tolist()
except Exception as e:
    print(f"[!] Warning: Models not found. You must run run_pipeline.py first! Error: {e}")
    rf_model = None
    scaler = None
    training_cols = []

def predict_fraud(transaction_data: dict) -> dict:
    """
    What it does: Translates JSON/Dict into a row, scales it, asks XGBoost to guess, returns human readable response.
    Why it is needed: Real time data doesn't come in large CSVs.
    Inputs: transaction_data dictionary
    Returns: A dictionary with results.
    """
    
    if rf_model is None or scaler is None:
        return {"error": "Models not loaded. Run pipeline first."}

    # 1. Convert simple dict to a Pandas DataFrame of 1 row
    # This simulates what we did in feature_engineering.py
    
    df = pd.DataFrame([transaction_data])
    
    # 2. Format columns exactly like feature_engineering.py did
    # For example, dropping IDs and doing dummy (one-hot) encoding
    if 'transaction_id' in df.columns:
        df = df.drop(columns=['transaction_id'])
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    if 'transaction_time' in df.columns:
        df = df.drop(columns=['transaction_time'])
        
    if 'merchant_category' in df.columns:
        # Create the dummy columns manually based on the training columns
        for col in training_cols:
            if col.startswith('merchant_category_'):
                cat = col.split('merchant_category_')[1]
                if df['merchant_category'].iloc[0] == cat:
                    df[col] = 1
                else:
                    df[col] = 0
        df = df.drop(columns=['merchant_category'])
        
    # Ensure all columns exist and are in the correct exact order as training
    for col in training_cols:
        if col not in df.columns:
            df[col] = 0  # Fill missing columns with 0
            
    df = df[training_cols]

    # 3. Scale the data using the EXACT SAME SCALER from preprocessing
    X_scaled = scaler.transform(df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=training_cols)
    
    # 4. Ask model to predict 0 or 1, and get the percentage certainty.
    prediction = int(rf_model.predict(X_scaled_df)[0])
    probability = float(rf_model.predict_proba(X_scaled_df)[0][1])

    # 5. Add a simple reasoning module (Risk Factors)
    risk_factors = []
    if transaction_data.get('transaction_velocity_24h', 0) > 3:
        risk_factors.append("High transaction velocity (>3 in 24h).")
    if transaction_data.get('transaction_amount', 0) > 1000:
        risk_factors.append("Unusually high transaction amount.")
    if transaction_data.get('is_international', 0) == 1:
        risk_factors.append("International transaction flag active.")
        
    reasoning = " ".join(risk_factors) if risk_factors else "No major explicit rules triggered, based purely on behavioral tree logic."

    return {
        "fraud_prediction": prediction,
        "fraud_probability": round(probability, 4),
        "risk_factors": reasoning
    }

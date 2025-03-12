"""
File: data/generate_sample_data.py
Purpose: Synthesizes a realistic dataset covering normal and fraudulent transactions.
Why this logic is separated: Because actual bank data contains PII (Personally Identifiable Information) and cannot be pushed to GitHub, we need a script to create a reproducible fake dataset.
Which file imports this: None directly. Used primarily from the terminal or test fixtures.
Pipeline Position: Step 0. Runs before the pipeline starts to provide raw input CSV.
Why it matters: Shows the interviewer that you understand realistic data synthesis (imbalanced target, velocity indicators, correlated characteristics).
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_synthetic_transactions(num_records: int = 15000) -> pd.DataFrame:
    """
    Creates a dataframe simulating realistic credit card behavior.
    Inputs: num_records (int) - Number of rows to generate.
    Output: pd.DataFrame representing raw data.
    Why it is needed: Provides raw material to train our XGBoost/Random Forest models.
    """
    np.random.seed(42)
    
    # Generate base attributes
    transaction_ids = [f"tx_{i:06d}" for i in range(num_records)]
    customer_ids = np.random.choice([f"c_{i:03d}" for i in range(100)], num_records) # Simulate repeated customers
    
    # 98% Normal transactions, 2% Fraud (Realistic Imbalance)
    fraud_labels = np.random.choice([0, 1], size=num_records, p=[0.98, 0.02])
    
    data = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(num_records):
        is_fraud = fraud_labels[i] == 1
        
        # Behavior simulation for Fraud vs Normal
        if is_fraud:
            amount = np.random.exponential(scale=1500) # Fraud tends to involve larger ticket sizes
            distance = np.random.exponential(scale=500) # Fraud often happens far from home
            declines = np.random.poisson(lam=3)         # Fraudsters try and fail often
            international = np.random.choice([True, False], p=[0.6, 0.4]) # High rate of international fraud
            card_pres = np.random.choice([True, False], p=[0.1, 0.9]) # Most fraud is online (Card Not Present)
        else:
            amount = np.random.exponential(scale=60) + 5
            distance = np.random.exponential(scale=10)
            declines = np.random.poisson(lam=0.1)
            international = np.random.choice([True, False], p=[0.05, 0.95])
            card_pres = np.random.choice([True, False], p=[0.7, 0.3])
            
        # Additional feature columns
        time_offset = np.random.randint(0, 30*24*60*60) # Random second within last 30 days
        txn_time = base_time + timedelta(seconds=time_offset)
        
        # Late night logic: 12 AM to 5 AM is high risk
        is_late_night = txn_time.hour < 5
        if is_fraud and np.random.rand() > 0.5:
            txn_time = txn_time.replace(hour=np.random.randint(0, 4)) # Push fraud to late night sometimes
            is_late_night = True
        
        data.append({
            "transaction_id": transaction_ids[i],
            "customer_id": customer_ids[i],
            "card_id": f"card_{np.random.randint(0, 200)}",
            "merchant_category": np.random.choice(["grocery", "electronics", "travel", "dining", "entertainment"]),
            "payment_channel": np.random.choice(["online", "pos", "atm"]),
            "transaction_amount": round(float(amount), 2),
            "transaction_time": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            "distance_from_home": round(float(distance), 2),
            "avg_amount_7d": round(float(amount * np.random.uniform(0.5, 1.5)), 2) if not is_fraud else round(float(amount * np.random.uniform(0.1, 0.5)), 2),
            "txn_count_1h": np.random.randint(5, 15) if is_fraud else np.random.randint(0, 3),
            "txn_count_24h": np.random.randint(15, 30) if is_fraud else np.random.randint(1, 8),
            "previous_declines_24h": declines,
            "time_since_last_txn_hours": round(np.random.uniform(0.1, 1.0), 2) if is_fraud else round(np.random.uniform(1, 48), 2),
            "is_international": international,
            "card_present": card_pres,
            "is_late_night": is_late_night,
            "fraud_label": fraud_labels[i]
        })
        
    df = pd.DataFrame(data)
    # Sort chronologically
    df = df.sort_values(by="transaction_time").reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Generating simulated transaction dataset...")
    target_dir = os.path.dirname(os.path.abspath(__file__))
    df = create_synthetic_transactions(15000)
    
    file_path = os.path.join(target_dir, "sample_transactions.csv")
    df.to_csv(file_path, index=False)
    print(f"Dataset successfully created at {file_path}")
    print(f"Fraud Rate: {df['fraud_label'].mean():.2%}")
    print(df.head())

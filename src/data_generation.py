"""
File Name: data_generation.py
Purpose: Generates a completely fake but realistic dataset of credit card transactions.
Why it exists: Because real bank data is private, we must simulate it to build the project.
Files that depend on this: src/config.py
Files that use this: run_pipeline.py
Inputs: None (pulls config settings)
Outputs: Creates a CSV file in data/raw/
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from src.config import NUM_TRANSACTIONS, RAW_DATA_PATH, RANDOM_STATE

def generate_synthetic_data():
    """
    What it does: Creates a DataFrame of fake transactions and injects fraud logic.
    Why it is needed: To act as the foundation for the entire ML pipeline.
    Inputs: None
    Returns: A Pandas DataFrame filled with data.
    """
    print("[*] Generating synthetic transaction data... this might take a moment.")
    
    # We must seed random generators so the outputs are identical every time you run it
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    # 1. Create basics for 50,000 transactions
    transaction_ids = [f"TXN_{i}" for i in range(NUM_TRANSACTIONS)]
    
    # Let's say we have 1,000 unique customers calling into our payment system
    customer_ids = [f"CUST_{np.random.randint(1, 1000)}" for _ in range(NUM_TRANSACTIONS)]
    
    # Generate spending amounts (most are small, some are huge)
    # Using an exponential distribution pushes most values towards 0, long tail upwards
    amounts = np.random.exponential(scale=50, size=NUM_TRANSACTIONS) + 2.0 
    
    # Generate random transaction times over the last 30 days
    start_time = datetime.now() - timedelta(days=30)
    times = [start_time + timedelta(minutes=random.randint(0, 30 * 24 * 60)) for _ in range(NUM_TRANSACTIONS)]
    
    # Sort the data by time (important for rolling/history features later)
    times.sort()

    categories = ['grocery', 'electronics', 'travel', 'restaurant', 'online_shopping', 'utility']
    merchant_cats = [random.choice(categories) for _ in range(NUM_TRANSACTIONS)]
    
    # is_international is a boolean 0 or 1
    is_intl = np.random.choice([0, 1], size=NUM_TRANSACTIONS, p=[0.95, 0.05]) # 5% are international

    # 2. Build the initial dataframe
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'customer_id': customer_ids,
        'transaction_amount': amounts,
        'transaction_time': times,
        'merchant_category': merchant_cats,
        'is_international': is_intl,
        'is_fraud': 0 # Default everyone to honest for now
    })

    # 3. Inject Fraud Logic
    # In real life, fraud isn't entirely random. It follows patterns.
    # We will purposely override 'is_fraud' to 1 for rows matching suspicious rules.
    
    fraud_indices = []
    
    for idx, row in df.iterrows():
        prob_fraud = 0.005 # Base probability is half a percent
        
        # Rule 1: High amount electronics are risky
        if row['transaction_amount'] > 500 and row['merchant_category'] in ['electronics', 'travel']:
            prob_fraud += 0.20
            
        # Rule 2: International transactions are slightly risky
        if row['is_international'] == 1:
            prob_fraud += 0.05
            
        # Rule 3: Extremely high amount is incredibly risky
        if row['transaction_amount'] > 3000:
            prob_fraud += 0.50
            
        # Roll a random number between 0 and 1. If it's less than our probability, it's fraud.
        if random.random() < prob_fraud:
            fraud_indices.append(idx)

    # Apply the fraud labels
    df.loc[fraud_indices, 'is_fraud'] = 1

    print(f"[*] Generated {len(df)} transactions.")
    print(f"[*] Total fraud cases generated: {len(fraud_indices)} ({(len(fraud_indices)/len(df))*100:.2f}%)")

    # Save it to the raw folder for the next script to pick up
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[*] Raw data safely saved to {RAW_DATA_PATH}\n")
    return df

if __name__ == "__main__":
    # If someone directly types `python data_generation.py`, run the function.
    generate_synthetic_data()

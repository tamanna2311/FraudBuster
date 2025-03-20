"""
File Name: feature_engineering.py
Purpose: Translates basic columns (time, amount) into intelligent features (velocity, previous spend).
Why it exists: A machine learning model doesn't understand time natively. If a customer buys 5 things in a minute, we must explicitly calculate that "velocity" for the model to see it.
Files that depend on this: src/config.py
Files that use this: run_pipeline.py
Inputs: The raw CSV created by data_generation.py
Outputs: An updated pandas DataFrame with missing values filled and new columns added.
"""

import pandas as pd
from src.config import RAW_DATA_PATH

def create_features(input_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    What it does: Adds behavioral features like velocity to every row.
    Why it is needed: These features heavily boost precision and recall because fraud looks like a sudden change in behavior.
    Inputs: input_df (The raw dataframe. If None, it loads it from disk).
    Returns: A DataFrame loaded with new columns ready for Preprocessing.
    """
    
    if input_df is None:
        print("[*] Loading raw data from disk for feature engineering...")
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        df = input_df.copy()
        
    print("[*] Engineering new features...")

    # Ensure transaction_time is a standard Python datetime format
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    
    # Sort by customer, then by time. 
    # This is critical so we can look "backwards" to see what a customer did previously.
    df = df.sort_values(by=['customer_id', 'transaction_time']).reset_index(drop=True)

    # 1. Feature: Time Since Last Transaction
    # If this is 0 seconds, it's very suspicious.
    df['time_since_last_txn'] = df.groupby('customer_id')['transaction_time'].diff().dt.total_seconds()
    
    # First transaction for a user will be NaN (missing). Fill it with a high number (safe)
    df['time_since_last_txn'] = df['time_since_last_txn'].fillna(999999) 

    # 2. Feature: Transaction Velocity (7 Days Rolling Count)
    # How many transactions did this user do in the last 7 days? Hacky/simple simulation.
    # Grouping by customer, expanding, and counting works well enough for simple models.
    df['user_txn_count'] = df.groupby('customer_id').cumcount()

    # 3. Feature: Spending Deviation 
    # Is this current amount way larger than what the user normally spends?
    # First calculate the average previous spend per user
    df['user_mean_spend'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: x.expanding().mean().shift())
    df['user_mean_spend'] = df['user_mean_spend'].fillna(df['transaction_amount']) # fill NaNs for the first transaction
    
    # How different is the current amount from their average?
    df['dev_from_mean'] = df['transaction_amount'] / (df['user_mean_spend'] + 1) # +1 prevents dividing by zero

    # Drop columns that a Machine Learning model cannot read directly (Strings and Dates)
    # We drop IDs here because model math won't work on 'CUST_123'
    df_clean = df.drop(columns=['transaction_id', 'customer_id', 'transaction_time'])
    
    # Get dummies basically converts categorical strings into 0 or 1 columns.
    # Example: 'merchant_category' becomes 'merchant_category_grocery' = 1, 'merchant_category_travel' = 0
    df_clean = pd.get_dummies(df_clean, columns=['merchant_category'], drop_first=True)

    print(f"[*] Feature engineering complete. Total columns: {len(df_clean.columns)}")
    return df_clean

if __name__ == "__main__":
    df_engineered = create_features()
    print(df_engineered.head())

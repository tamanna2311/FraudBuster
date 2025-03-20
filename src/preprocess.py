"""
File Name: preprocess.py
Purpose: Splits the data into Train/Test, scales large numbers, and balances the fraud classes.
Why it exists: ML models perform badly when one column has values [0-1] and another has [1000-5000]. Also, because fraud is <1% of data, the model will just guess 'Not Fraud' to get 99% accuracy if we don't duplicate the fraud rows (SMOTE).
Files that depend on this: src/config.py, src/utils.py
Files that use this: run_pipeline.py
Inputs: The engineered dataframe.
Outputs: X_train, X_test, y_train, y_test saved as CSVs, and a scaler.pkl object saved to disk.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.config import TEST_SIZE_PERCENT, RANDOM_STATE, SCALER_PATH
from src.config import TRAIN_X_PATH, TEST_X_PATH, TRAIN_y_PATH, TEST_y_PATH
from src.utils import save_object

def prepare_data(df: pd.DataFrame):
    """
    What it does: Executes the scaling, splitting, and SMOTE balancing.
    Why it is needed: Converts business logic data into clean math matrices for Algorithmic Training.
    Inputs: Dataframe with all engineered features.
    Returns: None (Saves outcomes to disk for the training script to find).
    """
    print("[*] Preprocessing data (Splitting, Scaling, and Handling Imbalance)...")
    
    # 1. Separate Features (X) from the Answer/Label (y)
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    # 2. Train / Test Split
    # We hide 20% of the data to test the model on later. The model will NEVER see this during training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE_PERCENT, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Scaling
    # StandardScaler transforms numbers so the average is 0 and standard deviation is 1.
    scaler = StandardScaler()
    
    # We 'fit' (learn the math) only on the training data, so we don't accidentally learn testing secrets.
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to pandas dataframe after scaling (Scaler outputs raw numpy arrays)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    print(f"[*] Original Training Distribution: \n{y_train.value_counts()}")

    # 4. Handle Imbalance with SMOTE
    # SMOTE creates fake (synthetic) rows of the minority class (Fraud) so that 
    # Fraud and Not Fraud have a 50/50 split in the training data.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print(f"[*] Balanced Training Distribution (After SMOTE): \n{y_train_balanced.value_counts()}")

    # 5. Save the preprocessing outputs
    X_train_balanced.to_csv(TRAIN_X_PATH, index=False)
    X_test_scaled.to_csv(TEST_X_PATH, index=False)
    y_train_balanced.to_csv(TRAIN_y_PATH, index=False)
    y_test.to_csv(TEST_y_PATH, index=False)

    # VERY IMPORTANT: Save the exact scaler object. 
    # The API will need this exact math file to scale incoming live data later!
    save_object(scaler, SCALER_PATH)
    
    print("[*] Preprocessing complete. Data ready for training.")

if __name__ == "__main__":
    print("Run this through run_pipeline.py instead.")

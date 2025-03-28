"""
File Name: train.py
Purpose: Trains two different Machine Learning models (Random Forest and XGBoost) on the balanced data.
Why it exists: This is the core "Brain" of the AI. It looks at the answers and the features and figures out the math.
Files that depend on this: src/config.py, src/utils.py
Files that use this: run_pipeline.py
Inputs: The X_train and y_train CSV files saved by preprocess.py
Outputs: Saved .pkl model files in the models/ directory.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception as e:
    print(f"[!] Warning: Could not import xgboost (often due to missing libomp on Mac). Skipping XGBoost. Error: {e}")
    XGB_AVAILABLE = False

from src.config import TRAIN_X_PATH, TRAIN_y_PATH, RANDOM_STATE, RF_MODEL_PATH, XGB_MODEL_PATH
from src.utils import save_object

def train_models():
    """
    What it does: Loads the training data, initiates the algorithms, and saves the trained models.
    Why it is needed: It produces the AI artifacts that our fast API will use to score transactions.
    Inputs: None
    Returns: None
    """
    print("[*] Loading training data...")
    X_train = pd.read_csv(TRAIN_X_PATH)
    y_train = pd.read_csv(TRAIN_y_PATH)
    
    # We will 'flatten' y_train so it's a 1D array, which models prefer.
    y_train = y_train.values.ravel()

    # 1. Train Random Forest
    print("[*] Training Random Forest Classifier...")
    # Random Forest builds 100 decision trees and votes on the final answer.
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    save_object(rf, RF_MODEL_PATH)

    # 2. Train XGBoost (If available)
    if XGB_AVAILABLE:
        print("[*] Training XGBoost Classifier...")
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
        xgb.fit(X_train, y_train)
        save_object(xgb, XGB_MODEL_PATH)
    else:
        print("[*] Skipping XGBoost training due to missing library.")
    
    print("[*] Model training completed successfully.")

if __name__ == "__main__":
    train_models()

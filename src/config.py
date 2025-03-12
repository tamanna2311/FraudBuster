"""
File Name: config.py
Purpose: Serves as the central place to hold all hardcoded paths, parameters, and settings.
Why it exists: If we move a file, or change the number of fake rows to generate, we only change it here. It prevents bugs and cleans up other code.
Files that use this: Almost every file (data_generation.py, train.py, etc.)
Inputs: None directly
Outputs: Variables imported by others
"""

import os

# Project root directory computation (one folder up from src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define sub-directories
DATA_dir = os.path.join(BASE_DIR, 'data')
RAW_DATA_dir = os.path.join(DATA_dir, 'raw')
PROCESSED_DATA_dir = os.path.join(DATA_dir, 'processed')
MODELS_dir = os.path.join(BASE_DIR, 'models')

# Create the directories if they don't exist when this file is imported
os.makedirs(RAW_DATA_dir, exist_ok=True)
os.makedirs(PROCESSED_DATA_dir, exist_ok=True)
os.makedirs(MODELS_dir, exist_ok=True)

# File Paths
RAW_DATA_PATH = os.path.join(RAW_DATA_dir, 'synthetic_transactions.csv')
TRAIN_X_PATH = os.path.join(PROCESSED_DATA_dir, 'train_X.csv')
TEST_X_PATH = os.path.join(PROCESSED_DATA_dir, 'test_X.csv')
TRAIN_y_PATH = os.path.join(PROCESSED_DATA_dir, 'train_y.csv')
TEST_y_PATH = os.path.join(PROCESSED_DATA_dir, 'test_y.csv')

# Model Artifact Paths
SCALER_PATH = os.path.join(MODELS_dir, 'scaler.pkl')
RF_MODEL_PATH = os.path.join(MODELS_dir, 'rf_model.pkl')
XGB_MODEL_PATH = os.path.join(MODELS_dir, 'xgb_model.pkl')

# Basic Configuration
NUM_TRANSACTIONS = 50000  # How many fake transactions to generate
RANDOM_STATE = 42         # Fixed seed so our results are reproducible (the same every time)
TEST_SIZE_PERCENT = 0.20  # We will use 20% of data to test the models

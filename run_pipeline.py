"""
File Name: run_pipeline.py
Purpose: The main entrypoint that runs the entire machine learning project from start to finish.
Why it exists: Makes executing the whole workflow a simple 1-click operation instead of running 5 files manually.
Files that depend on this: None
Files that use this: Directly used by the user via terminal (`python run_pipeline.py`)
Inputs: None directly
Outputs: Console logs showing the progress of the entire project.
"""

from src.data_generation import generate_synthetic_data
from src.feature_engineering import create_features
from src.preprocess import prepare_data
from src.train import train_models
from src.evaluate import evaluate_models

def main():
    """
    What it does: Calls all the scripts in the correct order.
    Why it is needed: Data must be generated BEFORE being engineered, BEFORE being scaled, BEFORE training. Order matters.
    """
    print("\n=======================================================")
    print("   Starting FraudBuster Machine Learning Pipeline")
    print("=======================================================\n")

    # Step 1: Generate Fake Data
    raw_df = generate_synthetic_data()

    # Step 2: Add intelligence (velocities, rollups)
    engineered_df = create_features(raw_df)

    # Step 3: Handle scaling and SMOTE imbalance
    prepare_data(engineered_df)

    # Step 4: Train AI Models
    train_models()

    # Step 5: Test and Print Metrics
    evaluate_models()

    print("\n=======================================================")
    print("   Pipeline Execution Finished Successfully!")
    print("   Models saved in /models/. Ready to run API.")
    print("=======================================================\n")

if __name__ == "__main__":
    main()

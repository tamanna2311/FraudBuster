# Architecture Explained

This document is a simple, file-by-file explanation of how the data flows from start to finish. If you are a beginner, read this top-to-bottom to understand how the system works.

## Complete Project Flow

The architecture is split into two distinct lifecycle phases:
1. **The Training Pipeline (Offline)**: Data is generated, features are engineered, and models are trained and saved.
2. **The Prediction API (Real-time)**: A web server loads the saved model and waits for a specific JSON message to make a live prediction.

Data flows from raw numbers to cleaned rows, to a mathematical equation (the model), and finally to predictions.

---

## File-by-File Breakdown

### 1. `src/config.py`
- **What it does:** Contains all hardcoded configuration like file paths (`models/xgboost.pkl`), dataset sizes, and random seeds.
- **Why it exists:** If we want to change where a model is saved, we only change it here rather than looking for it in 5 different files. Used constantly by other files.

### 2. `src/utils.py`
- **What it does:** Simple tools to load and save `joblib` (Pickle) files which is how we save Python objects to the hard drive.
- **Why it exists:** Keeps the code clean. Any script needing to save a model imports from `utils.py`.

### 3. `src/data_generation.py`
- **What it does:** Uses random numbers to create 50,000 fake transaction rows, saving it to `data/raw/synthetic_transactions.csv`.
- **Why it exists:** We don't have public bank data for privacy reasons. This simulates it, injecting simple fraud logic (like: if amount > 5000 and international, high chance of fraud).

### 4. `src/feature_engineering.py`
- **What it does:** Reads the raw CSV and calculates new, more complex columns.
- **Why it exists:** Giving a model simple data is bad. Giving it "derived" data like `transaction_velocity` (purchases in a short time frame) makes the model incredibly accurate.

### 5. `src/preprocess.py`
- **What it does:**
  - Standardizes the math: Converts large monetary values into scaled decimal variables so the math equations work better (Scaling).
  - Handles Imbalance: Uses SMOTE (Synthetic Minority Over-sampling Technique) to duplicate the few fraud records we have so the model can learn them equally.
- **Why it exists:** Most algorithms break if numbers are unscaled or if fraud is only 1% of the data.

### 6. `src/train.py`
- **What it does:** Takes the preprocessed training datasets and trains a RandomForest and XGBoost machine learning model.
- **Why it exists:** This is the core "AI" part. It learns the patterns and saves its "brain" to the `models/` directory using `utils.py`.

### 7. `src/evaluate.py`
- **What it does:** Loads the testing portion of the dataset, asks the models to guess, and compares their guesses to the real answers, outputting metrics like Precision and Recall.
- **Why it exists:** To prove the project works and to catch regressions if code changes.

### 8. `run_pipeline.py`
- **What it does:** The ringleader. It literally just imports all the above src files and runs them in order: Data Gen -> Features -> Preprocess -> Train -> Evaluate.
- **Why it exists:** So we can run the whole project with one command.

---

### The API Side

### 9. `api/schemas.py`
- **What it does:** Uses `pydantic` to define what a JSON Request is allowed to look like (e.g. `transaction_amount` must be a Float).
- **Why it exists:** If a bad application sends broken text instead of a number, the FastAPI server will cleanly reject it rather than crashing our model.

### 10. `src/predict.py`
- **What it does:** An engine specifically for the API. It takes one single dictionary row representing a transaction and runs it through the saved scaler and saved model.
- **Why it exists:** In production, you don't run 50,000 rows. You run 1 row real-fast. This script isolates that single row logic.

### 11. `api/main.py`
- **What it does:** Exposes the `predict.py` logic over a web REST API via `/predict`.
- **Why it exists:** This is how real engineering teams work. A payment gateway hits a web API, they don't load Python models directly into their code. This is the application front door.

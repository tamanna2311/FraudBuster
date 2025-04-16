# FraudBuster - Credit Card Fraud Detection

## 1. Project Overview
FraudBuster is an end-to-end, locally runnable Machine Learning pipeline designed to detect fraudulent credit card transactions in near real-time. It processes synthetic transaction data, extracts behavioral features (like transaction velocity), applies machine learning to identify anomalies, and serves predictions via a REST API.

## 2. Problem Statement
Credit card fraud costs the financial sector billions of dollars annually. To stop fraud, institutions need an automated system that flags suspicious transactions the moment they occur without rejecting too many legitimate transactions (false positives).

## 3. Why Fraud Detection Matters
A good fraud detection system protects consumers from financial loss and saves businesses from chargebacks. Because fraudulent transactions make up less than 1% of total transactions, identifying them requires specialized ML techniques handling high class imbalance.

## 4. Project Architecture
The project is built in multiple stages, creating a clear workflow from data to service:
1. **Data Generation:** Creates a synthetic dataset with realistic patterns.
2. **Feature Engineering:** Adds intelligence (like rolling averages and velocity).
3. **Preprocessing:** Scales numbers and balances the data using SMOTE.
4. **Model Training:** Trains RandomForest and XGBoost classifiers.
5. **Evaluation:** Checks Precision and Recall metrics.
6. **Inference API:** A FastAPI service loads the saved model for real-time scoring.

## 5. Folder Structure
```
FraudBuster/
├── data/
│   ├── raw/                 # Raw datasets
│   └── processed/           # Engineered data
├── models/                  # Saved .pkl models
├── src/
│   ├── data_generation.py   # Script to generate synthetic data
│   ├── feature_engineering.py# Creates transaction history features
│   ├── preprocess.py        # Scaling & Imbalance handling
│   ├── train.py             # ML Model Training
│   ├── evaluate.py          # Metrics (Precision, Recall)
│   ├── predict.py           # Loads model to test a single row
│   └── config.py            # Hardcoded paths and logic variables
├── api/
│   ├── main.py              # FastAPI Web Server
│   └── schemas.py           # Pydantic schemas for verification
├── tests/
├── run_pipeline.py          # Runs the entire ML generation flow
├── README.md                # This file
├── architecture_explained.md# Detailed file-by-file logic logic 
└── requirements.txt         # Dependencies
```

## 6. Step-by-step Execution Guide
**Step 1:** Install dependencies
```bash
pip install -r requirements.txt
```
**Step 2:** Run the entire Machine Learning pipeline
```bash
python run_pipeline.py
```
*Creates the dataset, trains the model, and evaluates it.*

**Step 3:** Start the prediction API
```bash
uvicorn api.main:app --reload
```
*Server will start running locally at http://127.0.0.1:8000*

## 7. Model Details
The system trains both a **RandomForest** classifier and an **XGBoost** classifier. These ensemble tree models are heavily favored in tabular finance data because they capture non-linear relationships quickly and deal with tabular features effectively.

## 8. Feature Engineering Explanation
Raw data (like amount and location) is rarely enough. The system engineers:
- **Transaction Velocity:** Count of transactions in the last 24 hours.
- **Unusual Spend Indicator:** If the current transaction is significantly larger than previous.
These behavioral insights vastly improve detection capabilities.

## 9. API Usage
The `FastAPI` service exposes a POST `/predict` endpoint that takes a JSON transaction payload, processes it the same way as training data, and returns a fraud probability.

## 10. Sample Request and Response
### Request
Send a POST request to `http://127.0.0.1:8000/predict`:
```json
{
  "transaction_id": "T12345",
  "customer_id": "C999",
  "transaction_amount": 1500.50,
  "transaction_time": "2025-04-16T12:00:00",
  "merchant_category": "electronics",
  "is_international": 1,
  "transaction_velocity_24h": 5
}
```

### Response
```json
{
  "transaction_id": "T12345",
  "fraud_prediction": 1,
  "fraud_probability": 0.89,
  "risk_factors": "High transaction_velocity_24h, high transaction_amount"
}
```

## 11. Expected Output Files
After running `run_pipeline.py`, you will see:
- `data/raw/synthetic_transactions.csv`
- `data/processed/train_X.csv`
- `models/xgboost_model.pkl`
- `models/scaler.pkl`

## 12. Future Improvements
- **Graph Features:** Linking users by IP or Device ID.
- **Cloud Deployment:** Packaging in a Docker container and hosting on AWS.
- **Real Database:** Replacing CSVs with PostgreSQL.

## 13. Interview Explanation Section

**How to explain this project in an interview:**
"I built an end-to-end Machine Learning pipeline to detect credit card fraud. Since getting real banking data is difficult, I simulated 50,000 transactions, injecting realistic fraud patterns. The core of my success wasn't just the model, but feature engineering—I created variables like 'transaction velocity'. I dealt with typical fraud class-imbalance using SMOTE. I trained an XGBoost model, achieving over 90% precision and a high recall, meaning we caught fraud without angering regular customers. Finally, I wrapped the model in a FastAPI REST endpoint to simulate how it would be used in a real payment architecture."

**Likely Interview Questions:**
- *Why did you use SMOTE?* Because fraud is less than 1% of data. Without SMOTE, the model would simply guess "Not Fraud" 99% of the time and look highly accurate, while failing the business goal completely.
- *Why Precision and Recall instead of Accuracy?* Accuracy is misleading on imbalanced data. Precision tells us "when we flagged fraud, were we right?" (minimizing customer anger). Recall tells us "of all real fraud, how much did we catch?" (minimizing financial loss).


# Project Flow Explained

The FraudBuster project is split into two distinct lifecycle phases: The **Offline Training Pipeline** and the **Online Inference API**.

You can think of it like building a car in a factory (Phase 1), and then later driving that car on the road (Phase 2). 

Here is the exact order in which the files execute:

---

## Phase 1: The Factory (Training Pipeline)
This happens when you run `python run_pipeline.py`. It executes these 5 scripts one-by-one in perfectly chronological order to build the Artificial Intelligence "Brain":

1. **`src/data_generation.py`** (The Raw Materials)
   * **What happens:** Generates 50,000 completely synthetic banking transactions and drops them into a raw CSV file. 
2. **`src/feature_engineering.py`** (Refining)
   * **What happens:** Picks up that raw data and calculates "smart" behavioral features, such as transaction velocity (how many times a user swiped their card in the last 24 hours).
3. **`src/preprocess.py`** (Polishing)
   * **What happens:** Our AI requires numbers to be clean mathematical fractions or else it breaks. It scales monetary values down and uses **SMOTE** to balance the fraud cases so the AI gets a fair look at both classes. It saves a `scaler.pkl` file for future use.
4. **`src/train.py`** (Building the Brain)
   * **What happens:** It feeds the clean data into Random Forest and XGBoost algorithms. The algorithms learn the fraud patterns and save their logic to the disk as `rf_model.pkl` and `xgb_model.pkl`. 
5. **`src/evaluate.py`** (Quality Control)
   * **What happens:** Opens up a hidden 20% of the dataset to test the newly minted models. It outputs the Precision and Recall metrics to your screen to prove it accurately detects fraud.

At this point, the factory work is done. `run_pipeline.py` finishes and turns off.

---

## Phase 2: On The Road (The REST API)
This happens when you start your web server using `uvicorn api.main:app`. The system is now waiting in near real-time for production transactions to occur.

1. **`api/main.py`** (The Front Door / Bouncer)
   * **What happens:** Someone swiping a credit card online sends an HTTP POST JSON payload holding their transaction values. 
2. **`api/schemas.py`** (The ID Checker)
   * **What happens:** Immediately intercepts the data to ensure nothing is corrupt (e.g., ensuring `transaction_amount` is an actual float number, not spoofed text).
3. **`src/predict.py`** (The Driver)
   * **What happens:** If the payload is entirely safe, it gets handed to the predictor. The predictor loads the `rf_model.pkl` brain that we built in Phase 1, scales the incoming row, and asks the brain: "Is this new transaction fraud?" It then returns the percentage probability securely back down the line to the user.

And that is the complete flow of data across the architecture! Everything relies entirely on running `run_pipeline.py` first so that `predict.py` has the saved artifacts it needs to execute later.

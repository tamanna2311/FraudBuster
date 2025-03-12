"""
File Name: utils.py
Purpose: Contains tiny helper functions to save and load Python objects to the hard drive.
Why it exists: We need to save the trained model. Re-writing "joblib.dump" and error handling everywhere is messy.
Files that depend on this: None
Files that use this: train.py, preprocess.py, predict.py
Inputs: Machine learning models or strings (paths)
Outputs: None (saves file) or returns Model object
"""

import joblib
import os

def save_object(obj, path):
    """
    What it does: Saves an object (like a trained AI model) to a file on your hard drive.
    Why it is needed: So we can reuse the model later in our API without retraining it for 10 minutes.
    Inputs: 
       - obj: The python object to save
       - path: Where to save it (e.g. models/model.pkl)
    Returns: None
    """
    joblib.dump(obj, path)
    print(f"[*] Object successfully saved to: {path}")

def load_object(path):
    """
    What it does: Reads a saved object from the hard drive back into Python.
    Why it is needed: So our API can load the model instantly upon starting.
    Inputs: path (Where the model is)
    Returns: The loaded python object.
    Raises: Exception if file doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find the file at {path}. Did you train the model?")
    return joblib.load(path)

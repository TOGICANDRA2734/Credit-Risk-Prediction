from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.utils.helper import load_joblib, load_params
from src.data_pipeline.data_defense import data_defense_checker
from src.preprocessing.preprocess import preprocess_process
import numpy as np

# init models and params
params = load_params(param_dir = "config/params.yaml")
best_model = load_joblib(path = params["model_dump_path"] + "xgb_best_model_v2.pkl")

# create FastAPI object
app = FastAPI()

# init base model to define the data type
class APIData(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: int
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# for root dir website, do this process
@app.get("/")
def root():
    return {
        "msg": "Hello",
        "status": "success"
    }


# service for predict ML model based on input data from user
@app.post("/predict")
def predict(data: APIData):
    
    # Convert input data to DataFrame
    df_data = pd.DataFrame([data.model_dump()])
 
    # Validate using data checker
    try:
        data_defense_checker(input_data=df_data, params=params)
    except AssertionError as ae:
        return {
            "res": [],
            "error_msg": str(ae),
            "status_code": 400
        }
        
    # If valid, preprocess the data
    df_data = preprocess_process(data=df_data, params=params)

    # Predict the input data
    y_pred = best_model.predict(df_data)

    try:
        prediction_value = float(y_pred[0])  # Convert to float
        # Format the number to be more readable
        formatted_prediction = round(prediction_value, 2)  # rounds to 2 decimal places
        
        return {
                "res": "Found API",
                "non_performing_loan_prediction": formatted_prediction,
                "status_code": 200,
                "error_msg": ""
            }
    except (IndexError, TypeError, ValueError) as e:
        return {
            "res": "Failed API",
            "non_performing_loan_prediction": None,
            "status_code": 500,
            "error_msg": f"Error processing prediction: {str(e)}"
        }

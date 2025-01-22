import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib


def ohe_fit_home_ownership(data_train: pd.DataFrame, column: str, params: dict):
    ohe_home_ownership = OneHotEncoder(categories=[params['value_home_ownership_status']], sparse_output=False)

    ohe_home_ownership.fit(data_train[[column]])
    
    joblib.dump(ohe_home_ownership, params["dataset_dump_path"]["processed"] + "ohe_home_ownership_fix.pkl")
    
    return ohe_home_ownership

def ohe_fit_loan_intent(data_train: pd.DataFrame, column: str, params: dict):
    ohe_loan_intent = OneHotEncoder(categories=[params['value_loan_intent_status']], sparse_output=False)

    ohe_loan_intent.fit(data_train[[column]])
    
    joblib.dump(ohe_loan_intent, params["dataset_dump_path"]["processed"] + "ohe_loan_intent_fix.pkl")
    
    return ohe_loan_intent

def ohe_fit_loan_grade(data_train: pd.DataFrame, column: str, params: dict):
    ohe_loan_grade = OneHotEncoder(categories=[params['value_loan_grade_status']], sparse_output=False)

    ohe_loan_grade.fit(data_train[[column]])
    
    joblib.dump(ohe_loan_grade, params["dataset_dump_path"]["processed"] + "ohe_loan_grade_fix.pkl")
    
    return ohe_loan_grade


def preprocess_ohe(data: pd.DataFrame, ohe, column) -> pd.DataFrame:
    ohe_feat = ohe.transform(data[[column]])

    # create dataframe
    ohe_cols = ohe.categories_[0]
    ohe_df = pd.DataFrame(ohe_feat, columns = ohe_cols, index = data.index)

    final_df = pd.concat([data, ohe_df], axis = 1)

    final_df = final_df.drop(columns = [column])

    return final_df

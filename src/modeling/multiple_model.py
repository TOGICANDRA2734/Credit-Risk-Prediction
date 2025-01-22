from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from src.utils.helper import dump_joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import pandas as pd


def modeling_multiple(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    svm_model = SVC(kernel='rbf', C=1, gamma="scale", random_state=42)
    
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    
    dump_joblib(rf_model, params["model_dump_path"] + "dt_baseline.pkl")
    dump_joblib(xgb_model, params["model_dump_path"] + "dt_baseline.pkl")
    dump_joblib(svm_model, params["model_dump_path"] + "dt_baseline.pkl")

    return rf_model, xgb_model, svm_model


# def modeling_linreg(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
#     linreg = LinearRegression()

#     linreg.fit(X_train, y_train)
    
#     dump_joblib(linreg, params["model_dump_path"] + "vanilla_linreg_model.pkl")
    
#     return linreg


# def predict_baseline(model, X_valid, y_valid):
#     y_pred_dummy = model.predict(X_valid)
    
#     print(f"MSE: {mean_squared_error(y_valid, y_pred_dummy)}")
#     print(f"R2: {r2_score(y_valid, y_pred_dummy)}")

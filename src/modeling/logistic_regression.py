from sklearn.linear_model import LogisticRegression
from src.utils.helper import dump_joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import pandas as pd


def modeling_logreg(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)
    
    dump_joblib(logreg, params["model_dump_path"] + "vanilla_logreg_model.pkl")
    
    return logreg


def predict_baseline(model, X_valid, y_valid):
    y_pred_logreg = model.predict(X_valid)
    
    accuracy = accuracy_score(y_valid, y_pred_logreg)
    precision = precision_score(y_valid, y_pred_logreg, average='weighted') 
    recall = recall_score(y_valid, y_pred_logreg, average='weighted')
    f_beta = fbeta_score(y_valid, y_pred_logreg, beta=1, average='weighted')

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F-beta Score: {f_beta:.3f}")

from sklearn.dummy import DummyClassifier
from src.utils.helper import dump_joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import pandas as pd


def modeling_baseline(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    dummy_regr = DummyClassifier(strategy='most_frequent')

    dummy_regr.fit(X_train, y_train)
    
    dump_joblib(dummy_regr, params["model_dump_path"] + "baseline_model.pkl")
    
    return dummy_regr


def predict_baseline(model, X_valid, y_valid):
    y_pred_dummy = model.predict(X_valid)
    
    accuracy = accuracy_score(y_valid, y_pred_dummy)
    precision = precision_score(y_valid, y_pred_dummy, average='weighted') 
    recall = recall_score(y_valid, y_pred_dummy, average='weighted')
    f_beta = fbeta_score(y_valid, y_pred_dummy, beta=1, average='weighted')

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F-beta Score: {f_beta:.3f}")

from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from src.utils.helper import load_joblib, dump_joblib


def hyperparam_process(model_path: str, X_train: pd.DataFrame, y_train: pd.Series):
    model = load_joblib(path = model_path)
    
    param_random = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [3, 6, 9, 12],
        "learning_rate": np.arange(0.01, 1.0, 0.01),
        "subsample": np.arange(0.6, 1.0, 0.05),
        "colsample_bytree": np.arange(0.6, 1.0, 0.05),
        "gamma": np.arange(0, 1, 0.1),
        "min_child_weight": np.arange(1, 10, 1)
    }
    
    k_folds = KFold(n_splits = 5)
    
    best_xgb_random = RandomizedSearchCV(estimator = model,
                                           param_distributions = param_random,
                                           cv = k_folds,
                                           verbose = 3,
                                           random_state=42,
                                        n_iter=50)
    
    best_xgb_random.fit(X_train, y_train)
    
    return best_xgb_random.best_params_


def best_model_train(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    best_xgb_tune = XGBClassifier(subsample=0.95,
            n_estimators=200,
            min_child_weight=4,
            max_depth=6,
            learning_rate=0.22,
            gamma=0.9,
            colsample_bytree=0.75
    )
    
    best_xgb_tune.fit(X_train, y_train)
    
    dump_joblib(best_xgb_tune, params["model_dump_path"] + "best_model.pkl")
    
    return best_xgb_tune


def predict_best_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Calculate all metrics
    accuracy_best = accuracy_score(y_test, y_pred)
    precision_best = precision_score(y_test, y_pred, average='weighted') 
    recall_best = recall_score(y_test, y_pred, average='weighted')
    f_beta_best = fbeta_score(y_test, y_pred, beta=1, average='weighted')

    # Print results
    print(f"Accuracy: {accuracy_best:.3f}")
    print(f"Precision: {precision_best:.3f}")
    print(f"Recall: {recall_best:.3f}")
    print(f"F-beta Score: {f_beta_best:.3f}")

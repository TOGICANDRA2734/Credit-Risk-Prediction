from sklearn.metrics import fbeta_score
from src.utils.helper import load_joblib


def test_model_performance():
    model = load_joblib("models/xgb_best_model_v2.pkl")
    
    X_test = load_joblib("data/processed/X_test_final.pkl")
    y_test = load_joblib("data/processed/y_test_final.pkl")
    
    y_pred = model.predict(X_test)
    
    fbeta = fbeta_score(y_test, y_pred, beta=1, average='weighted')
    
    THRESHOLD_FBETA_SCORE = 0.9
    
    assert fbeta > THRESHOLD_FBETA_SCORE, f"fbeta_score result is too high: {fbeta}"

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

test_input_data = {
    "person_age": 22,
    "person_income": 59000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "D",
    "loan_amnt": 35000,
    "loan_int_rate": 15.5,
    "loan_percent_income": 0.59,
    "cb_person_default_on_file": "Y",
    "cb_person_cred_hist_length": 3
}


def test_response_api():
    response = client.get("/")
    assert response.status_code == 200, "There's something wrong with the API"
    assert response.json() == {"msg":"Hello","status":"success"}, "Incorrect response"
    

def test_predict_api():
    response = client.post("/predict", json=test_input_data)
    print("Response:", response.text)  # This will show the error response if any
    print("Status code:", response.status_code)
    
    assert response.status_code == 200, "There's something wrong with the predict API"
    # response = client.post("/predict", json = test_input_data)
    
    # assert response.status_code == 200, "There's something wrong with the predict API"

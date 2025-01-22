# Model tested with XGBoost

# ML Process Pipeline
---

Step by step:
1. Business Derivation
2. Data Pipelines get from sources
3. EDA
4. Preprocessing
5. Modeling
6. Predict with API

### Predict with API
---

To predict, run command

```
fastapi dev app.py
```

Then, try this data to predict the house price using ML Model

```json
{
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
```

The output should be

```json
{
  "res": "Found API",
  "Non_loan_prediction": 1,
  "status_code": 200
}
```

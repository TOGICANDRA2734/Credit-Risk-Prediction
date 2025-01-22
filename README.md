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
  "area": 5000,
  "bedrooms": 3,
  "bathrooms": 4,
  "stories": 1,
  "mainroad": "yes",
  "guestroom": "yes",
  "basement": "no",
  "hotwaterheating": "yes",
  "airconditioning": "yes",
  "parking": 1,
  "prefarea": "no",
  "furnishingstatus": "furnished"
}
```

The output should be

```json
{
  "res": "Found API",
  "house_price_prediction": 8755880.38638432,
  "status_code": 200
}
```
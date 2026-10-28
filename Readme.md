# Titanic Survival Prediction API

## Overview
Predicts whether a Titanic passenger survived, using a scikit-learn
Random Forest pipeline served with FastAPI, containerized with Docker,
and tested via GitHub Actions CI.

- **Dataset:** Titanic (Kaggle)
- **Model:** RandomForestClassifier with preprocessing pipeline
- **Accuracy:** XX% on the test set
- **Training:** Google Colab (scikit-learn 1.6.1)

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs

## Run with Docker
```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

## Endpoints
- `GET /` health check
- `POST /predict` returns survival prediction

Example body:
```json
{"Pclass":1,"Sex":"female","Age":25,"SibSp":0,"Parch":0,"Fare":50,"Embarked":"S"}
```

## Tests and CI
Run `pytest -v`. GitHub Actions runs tests, builds the Docker image, and
smoke-tests the API on every push to `main`.
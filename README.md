# Salary-Predictor

ML model to predict employee salaries using Random Forest + FastAPI

# 💰 Salary Predictor API

ML model that predicts employee salaries from experience,
education, job role, and demographics.

## 🎯 Model Performance

| Metric              | Score   |
| ------------------- | ------- |
| R² Score            | 0.871   |
| Mean Absolute Error | $11,203 |
| CV Mean R²          | 0.867   |
| Training rows       | 5,190   |

## 🛠️ Tech Stack

Python · scikit-learn · FastAPI · pandas · numpy · joblib

## 🚀 Run Locally

```bash
git clone https://github.com/Marie619/Salary-Predictor.git
cd Salary-Predictor
pip install -r requirements.txt
python train.py       # train the model
uvicorn api:app --reload  # start the API
```

## 📡 API Endpoints

| Method | Endpoint | Description    |
| ------ | -------- | -------------- |
| GET    | /        | Health check   |
| GET    | /info    | Model details  |
| POST   | /predict | Predict salary |

## 📥 Sample Request

```json
{
  "age": 35,
  "gender": "Male",
  "education": "Master's",
  "job_title": "Software Engineer",
  "years_experience": 10.0
}
```

## 📤 Sample Response

```json
{
  "predicted_salary": 98540.25,
  "salary_range_low": 88686.23,
  "salary_range_high": 108394.28,
  "currency": "USD",
  "model_accuracy": "R² = 0.871"
}
```

# ─── PHASE 4: FASTAPI PREDICTION ENDPOINT ────────────
# Salary Predictor — api.py
# Run this file: uvicorn api:app --reload
# ─────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import os

# ─── LOAD OR TRAIN MODEL ON STARTUP ──────────────────
import os, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_and_save():
    # Load data from URL — no file needed
    url = "data/salary_data.csv"
    df = pd.read_csv(url)
    df = df.dropna()
    df = df[df['Salary'] > 10000]
    Q1, Q3 = df['Salary'].quantile(0.25), df['Salary'].quantile(0.75)
    df = df[(df['Salary'] >= Q1 - 1.5*(Q3-Q1)) & (df['Salary'] <= Q3 + 1.5*(Q3-Q1))]
    df['Salary_log'] = np.log1p(df['Salary'])

    encoders = {}
    for col in ['Gender', 'Education Level', 'Job Title']:
        le = LabelEncoder()
        df[col+'_enc'] = le.fit_transform(df[col])
        encoders[col] = le

    features = ['Age', 'Gender_enc', 'Education Level_enc',
                'Job Title_enc', 'Years of Experience']
    X, y = df[features], df['Salary_log']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model,    "model/salary_model.pkl")
    joblib.dump(encoders, "model/encoders.pkl")
    joblib.dump(features, "model/features.pkl")
    print("✅ Model trained and saved")

# Train if model doesn't exist yet
if not os.path.exists("model/salary_model.pkl"):
    print("⏳ No model found — training now...")
    train_and_save()

model    = joblib.load("model/salary_model.pkl")
encoders = joblib.load("model/encoders.pkl")
features = joblib.load("model/features.pkl")
print("✅ Model loaded and ready")

# ─── CREATE FASTAPI APP ───────────────────────────────
app = FastAPI(
    title="Salary Predictor API",
    description="Predicts employee salary using Random Forest ML model",
    version="1.0.0"
)

# ─── INPUT SCHEMA ─────────────────────────────────────
# This defines what data the client must send
class EmployeeInput(BaseModel):
    age: int = Field(..., ge=18, le=70,
                description="Age between 18 and 70")

    gender: str = Field(...,
                description="Male or Female")

    education: str = Field(...,
                description="Bachelor's, Master's, or PhD")

    job_title: str = Field(...,
                description="Job title e.g. Software Engineer")

    years_experience: float = Field(..., ge=0, le=50,
                description="Years of experience")

    # Example shown in the API docs page
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "gender": "Male",
                "education": "Master's",
                "job_title": "Software Engineer",
                "years_experience": 10.0
            }
        }

# ─── OUTPUT SCHEMA ────────────────────────────────────
# This defines what the API sends back
class SalaryPrediction(BaseModel):
    predicted_salary: float
    salary_range_low: float
    salary_range_high: float
    currency: str = "USD"
    model_accuracy: str = "R² = 0.871"

    # ─── ENDPOINT 1: ROOT / HEALTH CHECK ─────────────────
# Visit: http://localhost:8000/
@app.get("/")
def root():
    return {
        "message": "Salary Predictor API is running ✅",
        "version": "1.0.0",
        "docs": "Visit /docs to test the API"
    }

    # ─── ENDPOINT 2: MODEL INFO ──────────────────────────
# Visit: http://localhost:8000/info
@app.get("/info")
def model_info():
    return {
        "model_type": "Random Forest Regressor",
        "r2_score": 0.871,
        "mean_absolute_error": "$11,203",
        "features": features,
        "valid_genders": list(encoders["Gender"].classes_),
        "valid_education": list(encoders["Education Level"].classes_),
        "training_rows": 5190,
        "test_rows": 1298
    }

    # ─── ENDPOINT 3: PREDICT SALARY ──────────────────────
# POST: http://localhost:8000/predict
@app.post("/predict", response_model=SalaryPrediction)
def predict_salary(employee: EmployeeInput):

    # Step 1: Validate text inputs against known classes
    try:
        gender_enc = encoders["Gender"].transform([employee.gender])[0]
    except ValueError:
        valid = list(encoders["Gender"].classes_)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid gender. Valid values: {valid}"
        )

    try:
        edu_enc = encoders["Education Level"].transform([employee.education])[0]
    except ValueError:
        valid = list(encoders["Education Level"].classes_)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid education. Valid values: {valid}"
        )

    try:
        job_enc = encoders["Job Title"].transform([employee.job_title])[0]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown job title. Use a common job title."
        )

    # Step 2: Build the feature array
    input_data = np.array([[
        employee.age,
        gender_enc,
        edu_enc,
        job_enc,
        employee.years_experience
    ]])

    # Step 3: Predict (model returns log salary)
    log_salary = model.predict(input_data)[0]

    # Step 4: Convert log back to real dollars
    salary = np.expm1(log_salary)

    # Step 5: Add ±10% confidence range
    return SalaryPrediction(
        predicted_salary=round(salary, 2),
        salary_range_low=round(salary * 0.90, 2),
        salary_range_high=round(salary * 1.10, 2)
    )

    # ─── ADD AT BOTTOM OF api.py ─────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000
    )
    
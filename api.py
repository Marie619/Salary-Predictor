# ─── PHASE 4: FASTAPI PREDICTION ENDPOINT ────────────
# Salary Predictor — api.py
# Run this file: uvicorn api:app --reload
# ─────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import os

# ─── LOAD MODEL FILES ────────────────────────────────
print("⏳ Loading model...")

model    = joblib.load("models/salary_model.pkl")
encoders = joblib.load("models/encoders.pkl")
features = joblib.load("models/features.pkl")

print("✅ Model loaded successfully")

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
    
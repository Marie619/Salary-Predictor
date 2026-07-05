# ─── PHASE 5: STREAMLIT UI ───────────────────────────
# Salary Predictor — app.py
# Run: streamlit run app.py
# ─────────────────────────────────────────────────────

import streamlit as st
import requests
import json

import joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor
# ... (same train_and_save function from api.py)

# Load or train model directly in Streamlit
if not os.path.exists("model/salary_model.pkl"):
    train_and_save()

model    = joblib.load("model/salary_model.pkl")
encoders = joblib.load("model/encoders.pkl")
features = joblib.load("model/features.pkl")

# Then on button click — predict directly, no API call needed
if predict_btn:
    gender_enc = encoders["Gender"].transform([gender])[0]
    edu_enc    = encoders["Education Level"].transform([education])[0]
    job_enc    = encoders["Job Title"].transform([job_title])[0]

    input_data = np.array([[age, gender_enc, edu_enc, job_enc, experience]])
    log_salary  = model.predict(input_data)[0]
    salary      = np.expm1(log_salary)

    st.metric("💰 Predicted Salary", f"${salary:,.0f}")

# ─── PAGE CONFIG ─────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# ─── HEADER ──────────────────────────────────────────
st.title("💰 Employee Salary Predictor")
st.markdown("""
Predict an employee's expected salary based on their
experience, education, job role, and demographics.
Powered by a **Random Forest ML model** with **R² = 0.871**.
""")

# Horizontal divider
st.divider()

# ─── SIDEBAR INFO ────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this app")
    st.markdown("""
**Model:** Random Forest Regressor
**R² Score:** 0.871 \n
**Avg Error:** ±$11,203 \n
**Training data:** 5,190 employees + \n
**Built by:** Umair Majeed
    """)
    st.divider()
    st.caption("ML Engineering Portfolio Project")

    # ─── INPUT FIELDS ────────────────────────────────────
st.subheader("👤 Employee Details")

# Two columns side by side
col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age",
        min_value=18, max_value=70,
        value=30,
        help="Employee's current age"
    )
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"]
    )
    education = st.selectbox(
        "Education Level",
        options=["Bachelor's", "Master's", "PhD"]
    )

with col2:
    experience = st.slider(
        "Years of Experience",
        min_value=0, max_value=40,
        value=5,
        help="Total years of professional experience"
    )
    job_title = st.selectbox(
        "Job Title",
        options=[
            "Software Engineer", "Data Analyst",
            "Data Scientist", "Senior Engineer",
            "Manager", "Director",
            "Marketing Analyst", "Sales Associate",
            "HR Manager", "Financial Analyst",
            "Product Manager", "Project Manager",
            "Business Analyst", "Account Manager"
        ]
    )

st.divider()

# ─── PREDICT BUTTON ──────────────────────────────────
predict_btn = st.button(
    "🔮 Predict Salary",
    type="primary",
    use_container_width=True
)

if predict_btn:

    # Build the request payload
    payload = {
        "age": age,
        "gender": gender,
        "education": education,
        "job_title": job_title,
        "years_experience": float(experience)
    }

    # Call FastAPI and handle errors gracefully
    try:
        with st.spinner("Calculating salary..."):
            response = requests.post(API_URL, json=payload, timeout=10)
            result = response.json()

        if response.status_code == 200:

            # ── SUCCESS: Show results ──────────────────────
            st.success("✅ Prediction complete!")

            # Big salary number
            st.metric(
                label="💰 Predicted Annual Salary",
                value=f"${result['predicted_salary']:,.0f}"
            )

            # Salary range in 3 columns
            c1, c2, c3 = st.columns(3)
            c1.metric("Low Estimate",
                      f"${result['salary_range_low']:,.0f}")
            c2.metric("Predicted",
                      f"${result['predicted_salary']:,.0f}")
            c3.metric("High Estimate",
                      f"${result['salary_range_high']:,.0f}")

            st.divider()

            # Model info expander
            with st.expander("📊 See model details"):
                st.write(f"**Model accuracy:** {result['model_accuracy']}")
                st.write(f"**Avg prediction error:** ±$11,203")
                st.write("**Input sent to API:**")
                st.json(payload)

        else:
            st.error(f"API error: {result.get('detail', 'Unknown error')}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure FastAPI is running: uvicorn api:app --reload")
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")


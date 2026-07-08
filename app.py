# ─────────────────────────────────────────────────────
# Salary Predictor — app.py (Streamlit frontend)
# Run locally : streamlit run app.py
# Deploy      : push to GitHub → Streamlit Cloud
# ─────────────────────────────────────────────────────

import streamlit as st
import requests

# ─── API URL ─────────────────────────────────────────
# When running locally use localhost.
# When deployed on Streamlit Cloud use your Railway URL.
# Instructions: replace the Railway URL below with your
# actual deployed URL once Railway deployment is done.

import os
API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/predict"   # fallback for local dev
)

# ─── PAGE CONFIG — must be first Streamlit call ──────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# ─── DARK THEME CSS ──────────────────────────────────
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 820px;
    }
    [data-testid="metric-container"] {
        background: #1E2130;
        border: 0.5px solid #2E3250;
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #AFA9EC;
        font-size: 1.8rem;
        font-weight: 600;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #8B8FA8;
        font-size: 0.8rem;
    }
    .stButton > button {
        background: #534AB7;
        color: #FAFAFA;
        border: none;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 500;
        padding: 0.65rem 1.5rem;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #7F77DD;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] {
        background: #1E2130;
        border-right: 0.5px solid #2E3250;
    }
    hr { border-color: #2E3250; }
</style>
""", unsafe_allow_html=True)


# ─── HEADER ──────────────────────────────────────────
st.markdown("""
<div style="
    background:#1E2130;
    border:0.5px solid #2E3250;
    border-radius:12px;
    padding:1.5rem 1.75rem;
    margin-bottom:1.5rem;
    display:flex;
    align-items:center;
    gap:16px;
">
    <div style='font-size:2.8rem;'>💰</div>
    <div>
        <h1 style='margin:0;font-size:1.4rem;font-weight:600;color:#FAFAFA;'>
            Employee Salary Predictor
        </h1>
        <p style='margin:5px 0 0;font-size:0.85rem;color:#8B8FA8;'>
            Predict salary from experience, education &amp; job role
            &nbsp;·&nbsp; Random Forest ML &nbsp;·&nbsp; R&sup2; = 0.871
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.5rem 0 1rem;'>
        <div style='font-size:2.5rem;'>💰</div>
        <div style='font-weight:600;font-size:1rem;color:#FAFAFA;'>Salary Predictor</div>
        <div style='font-size:0.75rem;color:#8B8FA8;margin-top:4px;'>ML Engineering Portfolio</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    **Model:** Random Forest Regressor
    **R² Score:** 0.871
    **Avg Error:** ±$11,203
    **Training data:** 5,190 employees
    **Built by:** Umair Majeed
    """)
    st.divider()
    st.caption("ML Engineering Portfolio · 2025")


# ─── INPUT FIELDS ────────────────────────────────────
st.subheader("👤 Employee Details")
st.caption("Fill in the details and click Predict.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age",
        min_value=18, max_value=70, value=30,
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
        min_value=0, max_value=40, value=5,
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
    "🔮  Predict Salary",
    type="primary",
    use_container_width=True
)


# ─── RESULTS ─────────────────────────────────────────
if predict_btn:

    payload = {
        "age":              age,
        "gender":           gender,
        "education":        education,
        "job_title":        job_title,
        "years_experience": float(experience)
    }

    try:
        with st.spinner("⏳ Calculating salary prediction..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=60
            )
            result = response.json()

        if response.status_code == 200:
            st.success("✅ Prediction complete!")
            st.markdown("### 💰 Predicted Salary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Low Estimate",  f"${result['salary_range_low']:,.0f}")
            c2.metric("Predicted",     f"${result['predicted_salary']:,.0f}")
            c3.metric("High Estimate", f"${result['salary_range_high']:,.0f}")

            st.divider()

            with st.expander("📊 See model details"):
                st.write(f"**Model accuracy:** {result['model_accuracy']}")
                st.write("**Avg prediction error:** ±$11,203")
                st.write("**Input sent to API:**")
                st.json(payload)
        else:
            st.error(f"API error: {result.get('detail', 'Unknown error')}")

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to the API.\n\n"
            "If running locally: start FastAPI first with "
            "`uvicorn api:app --reload`\n\n"
            "If on Streamlit Cloud: set the API_URL secret "
            "in your Streamlit Cloud dashboard."
        )
    except requests.exceptions.Timeout:
        st.warning(
            "⏳ Request timed out. The API server may be "
            "waking up — please try again in 30 seconds."
        )
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")

else:
    st.markdown("""
    <div style="
        background:#1E2130;
        border:0.5px dashed #2E3250;
        border-radius:10px;
        padding:2.5rem;
        text-align:center;
    ">
        <div style='font-size:2.5rem;margin-bottom:10px;'>🎯</div>
        <div style='font-size:14px;font-weight:500;color:#FAFAFA;margin-bottom:6px;'>
            Fill in the details above and click Predict
        </div>
        <div style='font-size:12px;color:#8B8FA8;'>
            Your salary prediction will appear here
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── FOOTER ──────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;font-size:11px;color:#8B8FA8;padding:0.5rem 0;'>
    Built with Python · scikit-learn · Streamlit
    &nbsp;·&nbsp; Random Forest R&sup2; = 0.871
    &nbsp;·&nbsp;
    <a href='https://github.com/Marie619/Salary-Predictor'
       style='color:#7F77DD;text-decoration:none;'>
        View source on GitHub ↗
    </a>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Salary Predictor — app.py (Standalone, no external API)
# Run locally : streamlit run app.py
# Deploy      : push to GitHub → Streamlit Cloud
# ─────────────────────────────────────────────────────────────

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ─── MUST BE FIRST STREAMLIT CALL ────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────
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


# ─── MODEL TRAINING — CACHED ──────────────────────────────────
# @st.cache_resource means this runs ONCE then is reused
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    DATA_URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Salary_Data.csv"

    # Load and clean
    df = pd.read_csv(DATA_URL)
    df = df.dropna()
    df = df[df['Salary'] > 10000]

    Q1  = df['Salary'].quantile(0.25)
    Q3  = df['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    df  = df[
        (df['Salary'] >= Q1 - 1.5 * IQR) &
        (df['Salary'] <= Q3 + 1.5 * IQR)
    ]

    df['Salary_log'] = np.log1p(df['Salary'])

    # Encode text columns
    encoders = {}
    for col in ['Gender', 'Education Level', 'Job Title']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        encoders[col] = le

    # Features and target
    features = [
        'Age',
        'Gender_enc',
        'Education Level_enc',
        'Job Title_enc',
        'Years of Experience'
    ]
    X = df[features]
    y = df['Salary_log']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Metrics
    preds   = model.predict(X_test)
    r2      = r2_score(y_test, preds)
    mae     = mean_absolute_error(np.expm1(y_test), np.expm1(preds))

    metrics = {
        "r2"                  : round(r2, 3),
        "mae"                 : round(mae, 0),
        "n_train"             : len(X_train),
        "n_test"              : len(X_test),
        "feature_importances" : dict(zip(features, model.feature_importances_))
    }

    return model, encoders, features, metrics


# ─── LOAD MODEL (shows spinner while training) ────────────────
with st.spinner("⏳ Loading model — first visit takes ~30 seconds..."):
    model, encoders, features, metrics = load_or_train_model()


# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.5rem 0 1rem;'>
        <div style='font-size:2.5rem;'>💰</div>
        <div style='font-weight:600;font-size:1rem;color:#FAFAFA;'>Salary Predictor</div>
        <div style='font-size:0.75rem;color:#8B8FA8;margin-top:4px;'>ML Engineering Portfolio</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📊 Model Stats")
    st.metric("R² Score",   f"{metrics['r2']}")
    st.metric("Avg Error",  f"${metrics['mae']:,.0f}")
    st.metric("Trained on", f"{metrics['n_train']:,} employees")
    st.divider()
    st.markdown("""
    <div style='font-size:11px;color:#8B8FA8;line-height:1.8;'>
        <b>Model:</b> Random Forest<br>
        <b>Built by:</b> Umair Majeed<br>
        <b>Stack:</b> Python · scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("ML Engineering Portfolio · 2025")


# ─── HEADER ──────────────────────────────────────────────────
st.markdown(f"""
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
            &nbsp;·&nbsp; Random Forest
            &nbsp;·&nbsp; R&sup2; = {metrics['r2']}
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── INPUT FORM ──────────────────────────────────────────────
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
        options=list(encoders['Gender'].classes_)
    )
    education = st.selectbox(
        "Education Level",
        options=list(encoders['Education Level'].classes_)
    )

with col2:
    experience = st.slider(
        "Years of Experience",
        min_value=0, max_value=40, value=5,
        help="Total years of professional experience"
    )
    job_title = st.selectbox(
        "Job Title",
        options=sorted(list(encoders['Job Title'].classes_))
    )

st.divider()


# ─── PREDICT BUTTON ──────────────────────────────────────────
# Defined BEFORE any reference to predict_btn
predict_btn = st.button(
    "🔮  Predict Salary",
    type="primary",
    use_container_width=True
)


# ─── RESULTS — only runs when button is clicked ───────────────
if predict_btn:
    try:
        # Encode inputs
        gender_enc = encoders['Gender'].transform([gender])[0]
        edu_enc    = encoders['Education Level'].transform([education])[0]
        job_enc    = encoders['Job Title'].transform([job_title])[0]

        # Build feature array
        input_data = np.array([[
            age,
            gender_enc,
            edu_enc,
            job_enc,
            float(experience)
        ]])

        # Predict and reverse log transform
        log_salary = model.predict(input_data)[0]
        salary     = np.expm1(log_salary)
        low        = salary * 0.90
        high       = salary * 1.10

        # Show results
        st.success("✅ Prediction complete!")
        st.markdown("### 💰 Predicted Salary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Low Estimate",  f"${low:,.0f}")
        c2.metric("Predicted",     f"${salary:,.0f}")
        c3.metric("High Estimate", f"${high:,.0f}")

        st.divider()

        # Feature importance chart
        with st.expander("📊 What drives this prediction?", expanded=True):
            importance_data = metrics['feature_importances']
            labels = {
                'Age'                  : 'Age',
                'Gender_enc'           : 'Gender',
                'Education Level_enc'  : 'Education',
                'Job Title_enc'        : 'Job Title',
                'Years of Experience'  : 'Experience'
            }
            names  = [labels[f] for f in features]
            values = [importance_data[f] for f in features]

            sorted_pairs = sorted(zip(values, names))
            sorted_vals, sorted_names = zip(*sorted_pairs)

            fig, ax = plt.subplots(figsize=(7, 3))
            fig.patch.set_facecolor('#1E2130')
            ax.set_facecolor('#1E2130')
            ax.barh(sorted_names, sorted_vals,
                    color='#7F77DD', edgecolor='none', height=0.5)
            ax.set_xlabel('Importance Score', color='#8B8FA8', fontsize=10)
            ax.tick_params(colors='#FAFAFA', labelsize=10)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('#2E3250')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Input summary
        with st.expander("🔍 Your input summary"):
            summary = {
                "Age"                 : age,
                "Gender"              : gender,
                "Education"           : education,
                "Job Title"           : job_title,
                "Years of Experience" : experience,
                "Predicted Salary"    : f"${salary:,.0f}",
                "Salary Range"        : f"${low:,.0f} — ${high:,.0f}",
                "Model R²"            : metrics['r2']
            }
            for key, val in summary.items():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:13px;padding:5px 0;"
                    f"border-bottom:0.5px solid #2E3250;'>"
                    f"<span style='color:#8B8FA8;'>{key}</span>"
                    f"<span style='color:#FAFAFA;font-weight:500;'>{val}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
        st.info("Make sure all fields are filled correctly and try again.")

else:
    # Placeholder shown before prediction
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


# ─── FOOTER ──────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='text-align:center;font-size:11px;color:#8B8FA8;padding:0.5rem 0;'>
    Built with Python · scikit-learn · Streamlit &nbsp;·&nbsp;
    Random Forest R&sup2; = {metrics['r2']} &nbsp;·&nbsp;
    <a href='https://github.com/Marie619/Salary-Predictor'
       style='color:#7F77DD;text-decoration:none;'>
        View source on GitHub ↗
    </a>
</div>
""", unsafe_allow_html=True)

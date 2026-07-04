# Run this file: python train.py
# ─────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("✅ All libraries imported")

# Load the clean data from Phase 2
dir = "data/salary_clean.csv"
df = pd.read_csv(dir)
# Re-apply the same cleaning from Phase 2
df = df.dropna()
df = df[df['Salary'] > 10000]

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Salary'] >= Q1 - 1.5 * IQR) &
        (df['Salary'] <= Q3 + 1.5 * IQR)]

df['Salary_log'] = np.log1p(df['Salary'])

print(f"✅ Data loaded: {df.shape[0]} rows")

# ─── FEATURE ENCODING ────────────────────────────────
# Convert text columns to numbers using LabelEncoder

encoders = {}

text_columns = ['Gender', 'Education Level', 'Job Title']

for col in text_columns:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    encoders[col] = le    # save encoder for later use in API

print("✅ Encoding complete. New columns:")
for col in text_columns:
    print(f"   {col} → {col}_enc")
    print(f"   Classes: {list(encoders[col].classes_)}")

    # ─── DEFINE FEATURES AND TARGET ──────────────────────
features = [
    'Age',
    'Gender_enc',
    'Education Level_enc',
    'Job Title_enc',
    'Years of Experience'
]

X = df[features]           # input features
y = df['Salary_log']      # target (log-transformed salary)

# ─── SPLIT ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% goes to testing
    random_state=42      # fixed seed = same split every run
)

print(f"✅ Data split complete")
print(f"   Training rows : {X_train.shape[0]}")
print(f"   Testing rows  : {X_test.shape[0]}")
print(f"   Features used : {features}")

# ─── MODEL 1: LINEAR REGRESSION ──────────────────────
print("\n⏳ Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

lr_r2  = r2_score(y_test, lr_preds)
lr_mae = mean_absolute_error(
    np.expm1(y_test),
    np.expm1(lr_preds)   # convert back from log to dollars
)
print(f"   R² Score : {lr_r2:.3f}")
print(f"   MAE      : ${lr_mae:,.0f}")

# ─── MODEL 2: RANDOM FOREST ──────────────────────────
print("\n⏳ Training Random Forest (takes ~10 seconds)...")
rf = RandomForestRegressor(
    n_estimators=100,    # 100 decision trees
    max_depth=10,        # max depth of each tree
    random_state=42
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_r2  = r2_score(y_test, rf_preds)
rf_mae = mean_absolute_error(
    np.expm1(y_test),
    np.expm1(rf_preds)
)
print(f"   R² Score : {rf_r2:.3f}")
print(f"   MAE      : ${rf_mae:,.0f}")

# ─── CROSS VALIDATION ────────────────────────────────
print("\n⏳ Running 5-fold cross validation...")
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f"   CV Scores  : {[round(s,3) for s in cv_scores]}")
print(f"   Mean R²    : {cv_scores.mean():.3f}")
print(f"   Std Dev    : {cv_scores.std():.3f}")

# ─── FEATURE IMPORTANCE CHART ────────────────────────
importance = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values(ascending=True)

plt.figure(figsize=(8, 4))
importance.plot(kind='barh', color='#378ADD', edgecolor='none')
plt.title('Feature Importance — What drives salary?', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=150)
# plt.show()
print("\n✅ Feature importance chart saved to model/")
# ─── SAVE MODEL AND ENCODERS ─────────────────────────
# os.makedirs('model', exist_ok=True)

# Save the trained Random Forest model
joblib.dump(rf, 'models/salary_model.pkl')

# Save the encoders so API can convert text → numbers
joblib.dump(encoders, 'models/encoders.pkl')

# Save the feature list so API uses exact same order
joblib.dump(features, 'models/features.pkl')

print("\n✅ MODEL TRAINING COMPLETE!")
print("─" * 40)
print(f"   Model    : Random Forest")
print(f"   R² Score : {rf_r2:.3f}")
print(f"   CV Mean  : {cv_scores.mean():.3f}")
print
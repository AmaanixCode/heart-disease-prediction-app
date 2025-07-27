import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# Optional: Use XGBoost if available else will rely on logistic Regression
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    from sklearn.linear_model import LogisticRegression
    xgb_available = False

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: crimson;'>💓 Predict Heart Disease Risk</h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    data = pd.read_csv("heart_cleaned.csv")  # Ensure this file is present
    return data

df = load_data()

# ------------------ Sidebar Navigation ------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🧠 Predict", "📚 About"])

# ------------------ Train Model ------------------
def train_model():
    X = df.drop('target', axis=1)
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if xgb_available:
        base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        base_model = LogisticRegression()

    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, scaler, acc

model, scaler, accuracy = train_model()

# ------------------ Page: Dashboard ------------------
if page == "📊 Dashboard":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Target Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='target', palette="coolwarm", ax=ax1)
    ax1.set_title("Heart Disease (1: Yes, 0: No)")
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2)
    st.pyplot(fig2)

    st.success(f"✅ Calibrated Model Accuracy: **{accuracy * 100:.2f}%** using {'XGBoost' if xgb_available else 'Logistic Regression'}")

# ------------------ Page: Predict ------------------
elif page == "🧠 Predict":
    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.slider("Resting BP", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

    with col2:
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.slider("Max Heart Rate", 70, 210, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.50)

    input_df = pd.DataFrame([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]],
                            columns=df.columns[:-1])

    if st.button("Predict"):
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]
        if prob >= threshold:
            st.error(f"⚠️ High risk of heart disease. (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low risk of heart disease. (Probability: {prob:.2f})")

# ------------------ Page: About ------------------
else:
    st.markdown(f"""
    ### Project Overview
    This app uses a machine learning model trained on the **UCI Heart Disease Dataset** to predict whether a person is likely to have heart disease based on medical attributes.

    **Technologies Used**  
    - Python, Pandas, NumPy, Matplotlib, Seaborn  
    - Scikit-learn, XGBoost (optional), CalibratedClassifierCV  
    - Streamlit (for the dashboard)

    **Model Accuracy:** {accuracy * 100:.2f}%
    """)
    st.info("🔬 Built with ❤️ by Amaani for educational and practical data science learning.")

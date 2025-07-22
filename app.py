import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from utils import clean_strings


st.set_page_config(page_title="Salary Predictor", layout="wide")

MODEL_PATH = Path("salary_model.pkl")
DATA_PATH = Path("ds_sal.csv")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model not found. Train it using `train_model.py`.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.warning("Dataset not found.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()
    df["experience_level"] = df["experience_level"].str.lower()
    df["company_size"] = df["company_size"].str.lower()
    return df

# Load model and data
pipe = load_model()
df = load_data()

# App title
st.title("💰 Employee Salary Predictor")

tab1, tab2 = st.tabs(["🔮 Predict Salary", "📊 Visualize Dataset"])

# ---------------------------------------------------------------- #
# TAB 1: Prediction Interface
# ---------------------------------------------------------------- #
with tab1:
    st.subheader("Enter Employee Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        work_year = st.number_input("Work year", min_value=2015, max_value=2035, value=2023)
        experience_level = st.selectbox("Experience level", ["en", "mi", "se", "ex"])
        employment_type = st.selectbox("Employment type", ["pt", "ft", "ct", "fl"])
    with col2:
        job_title = st.text_input("Job title", value="data scientist")
        salary_currency = st.text_input("Salary currency (ISO)", value="usd")
        remote_ratio = st.slider("Remote ratio (%)", 0, 100, 100, step=10)
    with col3:
        employee_residence = st.text_input("Employee residence (ISO‑2)", value="us")
        company_location = st.text_input("Company location (ISO‑2)", value="us")
        company_size = st.selectbox("Company size", ["s", "m", "l"])

    if st.button("🔍 Predict Salary"):
        sample = pd.DataFrame({
            "work_year": [work_year],
            "experience_level": [experience_level],
            "employment_type": [employment_type],
            "job_title": [job_title],
            "salary_currency": [salary_currency],
            "employee_residence": [employee_residence],
            "remote_ratio": [remote_ratio],
            "company_location": [company_location],
            "company_size": [company_size]
        })
        prediction = pipe.predict(sample)[0]
        st.success(f"🎯 Estimated Salary: **${prediction:,.0f} USD**")

# ---------------------------------------------------------------- #
# TAB 2: Visualizations
# ---------------------------------------------------------------- #
with tab2:
    st.subheader("📊 Salary Analysis from Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💼 Average Salary by Job Title")
        top_jobs = (
            df.groupby("job_title")["salary_in_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=top_jobs, x="salary_in_usd", y="job_title", ax=ax1)
        ax1.set_xlabel("Average Salary (USD)")
        ax1.set_ylabel("Job Title")
        st.pyplot(fig1)

    with col2:
        st.markdown("### 🌍 Average Salary by Country")
        top_countries = (
            df.groupby("employee_residence")["salary_in_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=top_countries, x="salary_in_usd", y="employee_residence", ax=ax2)
        ax2.set_xlabel("Average Salary (USD)")
        ax2.set_ylabel("Country")
        st.pyplot(fig2)

    st.markdown("### 🧠 Salary vs Experience Level")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="experience_level", y="salary_in_usd", ax=ax3)
    ax3.set_xlabel("Experience Level")
    ax3.set_ylabel("Salary (USD)")
    st.pyplot(fig3)

    st.markdown("### 🔁 Remote Work Ratio vs Salary")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="remote_ratio", y="salary_in_usd", alpha=0.6, ax=ax4)
    ax4.set_xlabel("Remote Ratio (%)")
    ax4.set_ylabel("Salary (USD)")
    st.pyplot(fig4)

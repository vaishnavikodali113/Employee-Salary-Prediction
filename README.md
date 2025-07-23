# Employee Salary Prediction

This project applies machine learning techniques to predict employee salaries (in USD) based on job role, experience level, work location, and other employment-related features. It includes a trained regression model and a Streamlit web application for interactive salary prediction and data visualization.

---

## Project Overview

### Objective
To build an end-to-end machine learning solution that predicts salaries and deploys the model as an accessible web-based tool.

### Dataset
- Source: Kaggle - Data Science Job Salaries 2023  
  https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023
- Key features include:
  - `work_year`, `experience_level`, `employment_type`, `job_title`
  - `company_size`, `remote_ratio`, `employee_residence`, `company_location`

---

## Technologies Used

- Python 3.10+
- Scikit-learn for model building and pipeline integration
- Pandas and NumPy for data handling
- Joblib for model serialization
- Matplotlib, Seaborn, and Plotly for data visualization
- Streamlit for web app development and deployment

---

## Model Training

The model is a Random Forest Regressor trained within a preprocessing pipeline that handles:

- Ordinal encoding for `experience_level` and `company_size`
- One-hot encoding for categorical fields like `job_title` and `employment_type`
- Standard scaling for numeric fields like `work_year` and `remote_ratio`
- Custom string cleaning transformer

### Train the model:
```bash
python train_model.py --data_path ds_sal.csv --model_path salary_model.pkl

````markdown
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
````

### Evaluation Metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Streamlit Web Application

The Streamlit app allows users to:

* Input features and get real-time salary predictions
* Visualize salary trends with dynamic charts (e.g., sunburst chart by job and experience)

---

## Streamlit Deployment Guide

### Running the App Locally

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/employee-salary-prediction.git
   cd employee-salary-prediction
   ```

2. **Install Required Libraries**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

4. **View the App**

   Open your browser at:

   ```
   http://localhost:8501
   ```

---

### Deploying on Streamlit Cloud

1. **Push your project to GitHub**, including:

   * `app.py`
   * `salary_model.pkl`
   * `requirements.txt`
   * `ds_sal.csv` *(if used in visualizations)*

2. **Go to:** [https://streamlit.io/cloud](https://streamlit.io/cloud)

3. **Log in with GitHub** and click **"New App"**

4. **Configure repository and branch**, set entry point to:

   ```
   app.py
   ```

5. **Deploy** — You'll receive a public URL for sharing the app.

---

### Example `requirements.txt`

```txt
pandas
numpy
scikit-learn
joblib
streamlit
plotly
seaborn
matplotlib
```

---

## Project Structure

```
├── ds_sal.csv               # Dataset file
├── train_model.py           # Model training script
├── salary_model.pkl         # Trained pipeline/model
├── app.py                   # Streamlit application
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

---

## Future Enhancements

* Use XGBoost or LightGBM for improved regression performance
* Introduce embeddings for text-based inputs like job titles
* Support batch predictions via file upload
* Host on cloud platforms like Streamlit Cloud, AWS, or Hugging Face

---

## Acknowledgements

* IBM SkillsBuild & Edunet Foundation
* Kaggle and dataset contributors
* Streamlit and scikit-learn open source communities

---

## License

This project is intended for academic and non-commercial use only.

```

---

Let me know if you'd like this exported into a file (`README.md`) or want help writing a sample `app.py` or `requirements.txt`.
```

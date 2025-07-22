#!/usr/bin/env python3
"""
train_model.py
==============

Train a machine‑learning model that predicts employee salary in USD
from the ds_sal.csv dataset, and save the full preprocessing + model
pipeline to disk.

Usage
-----
$ python train_model.py \
    --data_path ds_sal.csv \
    --model_path salary_model.pkl
"""

from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)


# --------------------------------------------------------------------- #
# 1. Utility function used in both training and inference
# --------------------------------------------------------------------- #
def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and lowercase all object (string) columns.
    Returns the modified DataFrame (in‑place for pipelines via passthrough).
    """
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


# --------------------------------------------------------------------- #
# 2. Training function
# --------------------------------------------------------------------- #
def train(data_path: Path, model_path: Path) -> None:
    # ---- 2.1 Load data ------------------------------------------------ #
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Drop *local* salary column if present (not used for prediction)
    df.drop(columns=["salary"], errors="ignore", inplace=True)

    # ---- 2.2 Clean string columns ------------------------------------ #
    df = clean_strings(df)

    # ---- 2.3 Separate features & target ------------------------------ #
    y = df["salary_in_usd"]
    X = df.drop(columns=["salary_in_usd"])

    # ---- 2.4 Define column groups ------------------------------------ #
    ordinal_cols = {
        "experience_level": ["en", "mi", "se", "ex"],  # ordered
        "company_size": ["s", "m", "l"],
    }
    categorical_nominal = [
        "employment_type",
        "job_title",
        "salary_currency",
        "employee_residence",
        "company_location",
    ]
    numeric_cols = ["work_year", "remote_ratio"]

    # ---- 2.5 Build preprocessing ------------------------------------- #
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ord",
                OrdinalEncoder(categories=[ordinal_cols[c] for c in ordinal_cols]),
                list(ordinal_cols.keys()),
            ),
            (
                "nom",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_nominal,
            ),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    # ---- 2.6 Instantiate model --------------------------------------- #
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    # ---- 2.7 Combine into a pipeline --------------------------------- #
    pipe = Pipeline(
        steps=[
            ("pre_clean", FunctionTransformer(clean_strings)),  # ensures same cleaning at inference
            ("prep", preprocessor),
            ("rf", model),
        ]
    )

    # ---- 2.8 Train / evaluate ---------------------------------------- #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n─── Model Performance on 20 % Test Set ───")
    print(f"MAE  : ${mae:,.0f}")
    print(f"RMSE : ${rmse:,.0f}")
    print(f"R²   : {r2:.2f}")

    # ---- 2.9 Save pipeline ------------------------------------------- #
    joblib.dump(pipe, model_path)
    print(f"\n✅  Saved trained model →  {model_path.resolve()}")


# --------------------------------------------------------------------- #
# 3. CLI entry point
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train salary prediction model.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default="ds_sal.csv",
        help="Path to CSV dataset (default: ds_sal.csv)",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default="salary_model.pkl",
        help="Output path for the trained pipeline (default: salary_model.pkl)",
    )
    args = parser.parse_args()

    train(args.data_path, args.model_path)

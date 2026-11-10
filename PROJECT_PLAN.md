# 📋 Student Performance Prediction - Project Plan

## Project Overview

This project aims to build an end-to-end Machine Learning pipeline for predicting student academic performance using the Open University Learning Analytics Dataset (OULAD).

The project follows a complete production-style workflow including:

- Data Loading
- Data Cleaning
- Feature Engineering
- Target Engineering
- Preprocessing
- Model Training
- Benchmarking
- Hyperparameter Tuning
- Model Evaluation
- Explainability
- Inference Pipeline

---

# Objectives

## Classification Task

Predict the student's final result.

Target:

- Pass
- Fail
- Distinction
- Withdrawn

---

## Regression Task

Predict the student's final weighted score.

---

# Dataset

Open University Learning Analytics Dataset (OULAD)

Main Tables

- studentInfo
- studentAssessment
- assessments
- studentRegistration
- studentVle
- vle
- courses

---

# Project Architecture

```
Raw Data
    │
    ▼
Data Loader
    │
    ▼
Feature Engineering
    │
    ▼
Master Dataset
    │
    ▼
Preprocessing
    │
    ▼
Train/Test Split
    │
    ▼
Benchmark
    │
    ▼
Best Model
    │
    ▼
Evaluation
    │
    ▼
Artifacts
```

---

# Development Phases

## Phase 1

Project Structure

Status:

✅ Completed

---

## Phase 2

Data Loading

Status:

✅ Completed

---

## Phase 3

Feature Engineering

Implemented

- Registration Features
- Assessment Features
- Student VLE Features
- Activity Distribution
- Daily Statistics
- Session Statistics

Status:

✅ Completed

---

## Phase 4

Target Engineering

Implemented

- Classification Target
- Regression Target

Status:

✅ Completed

---

## Phase 5

Master Dataset Builder

Merged all features into one dataset.

Status:

✅ Completed

---

## Phase 6

Preprocessing

Implemented

- Missing Values
- Categorical Encoding
- Feature Scaling
- Train/Test Split

Status:

✅ Completed

---

## Phase 7

Model Benchmark

Classification Models

- Logistic Regression
- Decision Tree
- Random Forest

Regression Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Status:

✅ Completed

---

## Phase 8

Hyperparameter Tuning

Implemented using GridSearchCV.

Status:

✅ Completed

---

## Phase 9

Evaluation

Implemented

Classification

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Regression

- MAE
- RMSE
- MSE
- R²
- Residual Plot
- Prediction vs Actual

Status:

✅ Completed

---

## Phase 10

Feature Importance

Generated

- CSV Report
- Feature Importance Plot

Status:

✅ Completed

---

## Phase 11

Inference Pipeline

Implemented

- Load Saved Model
- Predict New Samples

Status:

✅ Completed

---

## Phase 12

Documentation

Included

- README
- Project Plan
- Requirements
- License

Status:

✅ Completed

---

# Outputs

Generated Files

```
outputs/

master_dataset.csv

classification_benchmark.csv

regression_benchmark.csv

plots/

classification_feature_importance.png

regression_feature_importance.png

confusion_matrix.png

prediction_vs_actual.png

residual_plot.png
```

---

# Artifacts

Saved Models

```
artifacts/

classification_model.pkl

regression_model.pkl
```

---

# Technologies

Programming Language

- Python

Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

# Project Metrics

Classification

- Accuracy
- Precision
- Recall
- F1

Regression

- MAE
- MSE
- RMSE
- R²

---

# Current Status

Project Version

v1.0

Current State

✅ Production Ready

---

# Future Improvements

- XGBoost
- LightGBM
- CatBoost
- SHAP Explainability
- FastAPI Deployment
- Docker Support
- CI/CD Pipeline
- MLflow Experiment Tracking
- Streamlit Dashboard

---

# Author

Abdelrahman salah

Faculty of Computer and Information Sciences

Ain Shams University

AI Engineer | Machine Learning | Generative AI
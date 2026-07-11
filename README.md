# House Price Prediction

## Overview

This project explores house price prediction using a log-transformed regression workflow. The repository includes the original exploratory notebook, a modular training/inference pipeline, and a static frontend for manual predictions.

## Problem Statement

The goal is to estimate house prices from property attributes such as area, room count, stories, parking availability, and road access. The analysis focuses on feature engineering, outlier handling, and regression modeling.

## Dataset

Dataset source: placeholder. The cleaned dataset used by the project is stored at [data/raw/Housing.csv](data/raw/Housing.csv).

## Technologies Used

- Python
- pandas
- numpy
- seaborn
- matplotlib
- plotly
- scikit-learn
- joblib
- HTML, CSS, JavaScript

## Project Structure

```
project-root/
├── data/
│   └── raw/
├── notebooks/
├── src/
├── images/
├── WEBSITE/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Workflow

1. Load the housing dataset from `data/raw/Housing.csv`.
2. Perform cleaning, outlier removal, and feature engineering.
3. Train the regression model on log-transformed price features.
4. Evaluate with a train/test split and repeated K-fold cross-validation.
5. Save the trained model and scaler artifacts for reuse.

## Results

The project’s reported performance is consistent with a mid-range regression fit on the transformed target, with test and cross-validation scores shown in the notebook and training script outputs. The model coefficients and evaluation plots remain part of the analysis.

## Future Improvements

- Add automated data validation before training.
- Split preprocessing and modeling into a formal pipeline object.
- Add lightweight tests for inference and feature preparation.
- Package the workflow for repeatable local execution.

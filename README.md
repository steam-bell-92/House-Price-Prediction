# 🏠 House Price Predictor

A simple, interactive machine learning project that predicts house prices using a multivariate linear regression model enhanced with ElasticNet regularization. The model is deployed on a static website where users can input values and receive real-time predictions based on the trained expression.

⭐ If you find this useful, consider giving it a star!

---

## 📌 Project Highlights

- Built using **`Python` (`scikit-learn`)** and essential data libraries
- Performed **EDA, feature engineering**, and **`IQR-based outlier removal`**
- Model trained using **`ElasticNet` `Regression`**
- Validated using **10×10 `Repeated K-Fold Cross Validation`**
- Achieved **`R² ≈ 0.57`** initially.
- Achieved **`R² ≈ 0.54`** after cross validation (10 folds).
- Website built with **`HTML`, `CSS`, and `JavaScript`**
- **Static website** provides instant predictions from model equation
  <br><br>
⭐ If you find this useful, consider giving it a star!

<img src="https://github.com/steam-bell-92/House-Price-Prediction/blob/main/Model.png">

---

## 🔍 Features

| 📊 EDA | 📉 Outlier & Feature Engineering | 🧠 ElasticNet Model | 🚀 Live Prediction |
|-------|------------------------------|-------------------|-----------------|
| Analyzed patterns using seaborn & matplotlib | Removed outliers (IQR) & added new features | Combines Lasso & Ridge | Input → Predict in browser |

---

## 🌐 Live Website
Try predicting prices of various houses: <a href='https://steam-bell-92.github.io/House-Price-Prediction/Housing_front.html'>WEBSITE</a><br>
> ***Caution***: dataset from 2022-23

---

## 🧰 Tech Stack

- **`Python`** – core language for ML
- **`pandas`**, **`numpy`** – data handling
- **`seaborn`**, **`matplotlib`**, **`plotly`** – visualization
- **`scikit-learn`** – model training and validation
- **`HTML/CSS`** – frontend structure and styling of website
- **`JavaScript`** – interactive logic for predictions

---

## 📁 Project Structure

```
House-Price-Prediction/
|
├── Codes/
│   ├── Housing.ipynb                🔹 ML Workflow
│   ├── housing.py                   🔹 Python File (same as ML Workflow)
│   └── Housing.csv                  🔹 Dataset
|
├── Images/
│   ├── Houses_prices_pic.jpg        🔹 Background on Website
│   └── Model.png                    🔹 Model Expression Plot
|
├── Website/
│   ├── Housing.html                 🔹 Main Site where Prediction
│   ├── Housing_front.html           🔹 Front Page of Website
│   ├── Housing_style.css            🔹 Main Site Styling
│   ├── Housing_front_style.css      🔹 Front Page Styling
│   └── Housing_script.js            🔹 Logic of Prediction
|
├── README.md                        🔹 This File !!
|
└── LICENSE                          🔹 MIT License

```
---

## 👤 Author
Anuj Kulkarni - aka - steam-bell-92

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

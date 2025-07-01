# 🏠 House Price Predictor

A simple, interactive machine learning project that predicts house prices using a multivariate linear regression model enhanced with ElasticNet regularization. The model is deployed on a static website where users can input values and receive real-time predictions based on the trained expression.

## 📌 Project Highlights

- Built using **`Python` (`scikit-learn`)** and essential data libraries
- Performed **EDA, feature engineering**, and **`IQR-based outlier removal`**
- Model trained using **`ElasticNet` `Regression`**
- Validated using **10×10 `Repeated K-Fold Cross Validation`**
- Achieved **`R² ≈ 0.57`** initially.
- Achieved **`R² ≈ 0.54`** after cross validation (10 folds).
- Website built with **`HTML`, `CSS`, and `JavaScript`**
- **Static website** provides instant predictions from model equation

## 🔍 Features

| 📊 EDA | 📉 Outlier & Feature Engineering | 🧠 ElasticNet Model | 🚀 Live Prediction |
|-------|------------------------------|-------------------|-----------------|
| Analyzed patterns using seaborn & matplotlib | Removed outliers (IQR) & added new features | Combines Lasso & Ridge | Input → Predict in browser |

## 🌐 Live Website
Try predicting prices of various houses: <a href='https://steam-bell-92.github.io/House-Price-Prediction/Housing_front.html'>WEBSITE</a><br>
> ***Caution***: dataset from 2022-23

## 🧰 Tech Stack

- **`Python`** – core language for ML
- **`pandas`**, **`numpy`** – data handling
- **`seaborn`**, **`matplotlib`** – visualization
- **`scikit-learn`** – model training and validation
- **`HTML/CSS`** – frontend structure and styling of website
- **`JavaScript`** – interactive logic for predictions

## 📁 Project Structure

```
House-Price-Predicton/
├── Housing_front.html                   🔹 Landing/intro page
├── Housing.html                         🔹 Prediction interface
├── Housing_style.css                    🔹 CSS file 
├── Housing_front_style.css              🔹 CSS file
├── Housing_script.js                    🔹 JS model logic (was: Housing_script.js)
├── Houses_prices_pic.jpg                🔹 Project image (was: Houses_prices_pic.jpg) 
├── Housing.csv                          🔹 Project Dataset (from: kaggle)    
├── Housing.ipynb                        🔹 Model training notebook
└── README.md                            🔹 This file !
```

## 👤 Author
Anuj Kulkarni - aka - steam-bell-92

# 🏠 House Price Predictor

A simple, interactive machine learning project that predicts house prices using a multivariate linear regression model enhanced with ElasticNet regularization. The model is deployed on a static website where users can input values and receive real-time predictions based on the trained expression.

## 📌 Project Highlights

- Built using **Python (scikit-learn)** and essential data libraries
- Performed **EDA, feature engineering**, and **IQR-based outlier removal**
- Model trained using **ElasticNet Regression**
- Validated using **10×10 Repeated K-Fold Cross Validation**
- Achieved **R² ≈ 0.57**, ~0.54 after cross-validation
- Frontend built with **HTML, CSS, and JavaScript**
- **Static website** provides instant predictions from model equation

## 🔍 Features

| 📊 EDA | 📉 Outlier & Feature Engineering | 🧠 ElasticNet Model | 🚀 Live Prediction |
|-------|------------------------------|-------------------|-----------------|
| Analyzed patterns using seaborn & matplotlib | Removed outliers (IQR) & added new features | Combines Lasso & Ridge | Input → Predict in browser |

## 🌐 Live Website
Try predicting prices of various houses: <a href='https://steam-bell-92.github.io/House-Price-Prediction/Housing_front.html'>WEBSITE</a>

## 🧰 Tech Stack

- **Python** – core language for ML
- **pandas**, **numpy** – data handling
- **seaborn**, **matplotlib** – visualization
- **scikit-learn** – model training and validation
- **HTML/CSS** – frontend structure and styling of website
- **JavaScript** – interactive logic for predictions

## 📁 Project Structure

```
house-price-predictor/
│
├── index.html                   # 🔹 Landing/intro page (was: Housing_front.html)
├── predictor.html               # 🔹 Prediction interface (was: Housing.html)
│
├── style/
│   └── main.css                 # 🔹 Combined CSS file (from Housing_front_style.css & Housing_style.css)
│
├── script/
│   └── model.js                 # 🔹 JS model logic (was: Housing_script.js)
│
├── assets/
│   ├── house_price_banner.jpg   # 🔹 Project image (was: Houses_prices_pic.jpg)                    
│   └── Housing.csv              # 🔹 Project Dataset (from: kaggle)    
|
├── notebook/
│   └── house_price_model.ipynb  # 🔹 Model training notebook (was: Housing.ipynb)
│
└── README.md                    # 🔹 Project documentation
```

## 👤 Author
Anuj Kulkarni - aka - steam-bell-92

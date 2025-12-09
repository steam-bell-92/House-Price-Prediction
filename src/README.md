# Machine Learning Pipeline - House Price Prediction

This directory contains the refactored machine learning pipeline for the House Price Prediction project. The code has been modularized from the original Jupyter notebook into separate, maintainable Python files.

## 📁 File Structure

```
src/
├── __init__.py          # Package initialization
├── config.py            # Configuration (hyperparameters, paths)
├── preprocessing.py     # Data preprocessing functions
├── train.py            # Model training script
└── inference.py        # Inference/prediction functions
```

## 🚀 Quick Start

### 1. Training the Model

To train the model and save artifacts:

```bash
cd src
python3 train.py
```

This will:
- Load and preprocess the data from `CODES/Housing.csv`
- Remove outliers using IQR method
- Engineer features (log transformations, etc.)
- Train an ElasticNet model
- Evaluate on test set and perform cross-validation
- Save the trained model and scaler to `models/` directory

### 2. Making Predictions

You can use the inference module in two ways:

**A. As a command-line tool:**
```bash
cd src
python3 inference.py
```

**B. Importing in your code:**
```python
from src.inference import make_prediction

# Make a single prediction
price = make_prediction(
    area=3000,           # square feet
    total_rooms=5,       # bedrooms + bathrooms
    stories=2,
    has_parking=1,       # 1 or 0
    mainroad=1          # 1 or 0
)

print(f"Predicted Price: ₹{price:,.2f}")
```

**C. Batch predictions:**
```python
from src.inference import batch_predict
import pandas as pd

data = pd.DataFrame({
    'area': [2500, 4000, 6000],
    'total_rooms': [4, 6, 8],
    'stories': [1, 2, 3],
    'has_parking': [0, 1, 1],
    'mainroad': [1, 1, 1]
})

prices = batch_predict(data)
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

### Model Settings
```python
MODEL_TYPE = 'elasticnet'  # or 'linear' for LinearRegression
ALPHA = 0.001             # Regularization strength (ElasticNet only)
L1_RATIO = 0.5            # ElasticNet mixing (0=Ridge, 1=Lasso)
RANDOM_STATE = 42         # For reproducibility
```

### File Paths
```python
DATA_PATH = 'path/to/Housing.csv'
MODEL_SAVE_PATH = 'path/to/save/model.pkl'
SCALER_SAVE_PATH = 'path/to/save/scaler.pkl'
```

## 📊 Model Features

The model uses the following features:
- `log_area`: Log-transformed house area
- `total_rooms`: Total number of rooms (bedrooms + bathrooms)
- `stories`: Number of stories
- `has_parking`: Binary indicator (1 if parking available)
- `mainroad`: Binary indicator (1 if on main road)

Target: `log_prices` (log-transformed house price)

## 🔧 Requirements

```bash
pip install numpy pandas scikit-learn joblib
```

## 📈 Model Performance

Current model achieves:
- **Test R² Score:** ~0.58
- **Cross-Validation Avg R²:** ~0.53 ± 0.09
- **Test RMSE:** ~0.25 (on log scale)

## 🔄 Data Preprocessing Pipeline

1. **Load Data**: Read CSV file
2. **Remove Outliers**: IQR method on price, area, rooms, and derived metrics
3. **Feature Engineering**:
   - Create `total_rooms` = bedrooms + bathrooms
   - Create `has_parking` from parking count
   - Convert categorical variables (yes/no → 1/0)
   - Apply log transformations to price and area
4. **Train-Test Split**: 80/20 split with fixed random state

## 🎯 Web App Integration

The web app can use the saved artifacts:

```python
# In your web app (e.g., Flask/Django)
from src.inference import make_prediction

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    price = make_prediction(
        area=data['area'],
        total_rooms=data['total_rooms'],
        stories=data['stories'],
        has_parking=data['has_parking'],
        mainroad=data['mainroad']
    )
    return jsonify({'predicted_price': price})
```

## 📝 Notes

- The scaler is saved but not currently used in predictions (log transformations handle scaling)
- The scaler is available for future enhancements if needed
- Model artifacts are saved in `models/` directory (excluded from git)
- Random state is fixed for reproducibility

## 🛠️ Development

To retrain with different parameters:
1. Edit `src/config.py`
2. Run `python3 src/train.py`
3. Test with `python3 src/inference.py`

## 📄 License

MIT License (same as parent project)

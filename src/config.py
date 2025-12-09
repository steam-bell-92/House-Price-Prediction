"""
Configuration file for the House Price Prediction model.
Contains hyperparameters and file paths.
"""
import os

# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ Hyperparameters ============
# ElasticNet hyperparameters
ALPHA = 0.1  # Regularization strength
L1_RATIO = 0.5  # ElasticNet mixing parameter (0=Ridge, 1=Lasso)

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42  # Fixed for reproducibility

# Cross-validation parameters
N_SPLITS = 10
N_REPEATS = 10

# ============ File Paths ============
# Data paths
DATA_PATH = os.path.join(PROJECT_ROOT, 'CODES', 'Housing.csv')

# Model artifacts paths
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'house_price_model.pkl')
SCALER_SAVE_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# ============ Feature Configuration ============
# Features used for prediction
FEATURE_COLUMNS = ['log_area', 'total_rooms', 'stories', 'has_parking', 'mainroad']
TARGET_COLUMN = 'log_prices'

# Original input features that need transformation
RAW_FEATURE_COLUMNS = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'mainroad']

# Categorical columns that need encoding (yes/no -> 1/0)
CATEGORICAL_COLUMNS = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                       'prefarea', 'airconditioning', 'has_parking']

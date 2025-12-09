"""
Data preprocessing functions for the House Price Prediction model.
Handles data cleaning, feature engineering, and scaler management.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import config


def load_data(data_path=None):
    """
    Load the housing dataset from CSV file.
    
    Args:
        data_path (str, optional): Path to the CSV file. Uses config.DATA_PATH if None.
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if data_path is None:
        data_path = config.DATA_PATH
    
    df = pd.read_csv(data_path)
    return df


def remove_outliers_iqr(df):
    """
    Remove outliers using the IQR (Interquartile Range) method.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    df_clean = df.copy()
    
    # Calculate total_rooms and price metrics first
    df_clean['total_rooms'] = df_clean['bedrooms'] + df_clean['bathrooms']
    df_clean['price/sq.feet'] = df_clean['price'] / df_clean['area']
    df_clean['price/rooms'] = df_clean['price'] / df_clean['total_rooms']
    
    # Columns to check for outliers
    outlier_columns = ['price', 'area', 'total_rooms', 'price/sq.feet', 'price/rooms']
    
    # Apply IQR method iteratively
    for col in outlier_columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)].copy()
    
    # Reset index
    df_clean.index = range(len(df_clean))
    
    return df_clean


def engineer_features(df):
    """
    Create engineered features for the model.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    df_processed = df.copy()
    
    # Create total_rooms if not already exists
    if 'total_rooms' not in df_processed.columns:
        df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['bathrooms']
    
    # Create has_parking feature
    df_processed['has_parking'] = 'no'
    df_processed.loc[df_processed['parking'] > 0, 'has_parking'] = 'yes'
    
    # Convert categorical columns from yes/no to 1/0
    def convert_str_to_int(col_name):
        df_processed[col_name] = df_processed[col_name].map({'yes': 1, 'no': 0, True: 1, False: 0})
    
    for col in config.CATEGORICAL_COLUMNS:
        if col in df_processed.columns:
            convert_str_to_int(col)
    
    # Create log transformations
    df_processed['log_prices'] = np.log(df_processed['price'])
    df_processed['log_area'] = np.log(df_processed['area'])
    
    return df_processed


def prepare_features(df):
    """
    Prepare features for training by selecting the required columns.
    
    Args:
        df (pd.DataFrame): Processed dataframe
    
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].copy()
    
    return X, y


def fit_scaler(X_train):
    """
    Fit a StandardScaler on the training data.
    Note: The current model uses log-transformed features which don't require scaling.
    This function is provided for future use if needed.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features
    
    Returns:
        StandardScaler: Fitted scaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler, save_path=None):
    """
    Save the fitted scaler to disk using joblib.
    
    Args:
        scaler: Fitted scaler object
        save_path (str, optional): Path to save the scaler. Uses config.SCALER_SAVE_PATH if None.
    """
    if save_path is None:
        save_path = config.SCALER_SAVE_PATH
    
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to: {save_path}")


def load_scaler(load_path=None):
    """
    Load a fitted scaler from disk.
    
    Args:
        load_path (str, optional): Path to load the scaler from. Uses config.SCALER_SAVE_PATH if None.
    
    Returns:
        Scaler object
    """
    if load_path is None:
        load_path = config.SCALER_SAVE_PATH
    
    scaler = joblib.load(load_path)
    return scaler


def preprocess_pipeline(data_path=None, remove_outliers=True):
    """
    Complete preprocessing pipeline from loading to feature preparation.
    
    Args:
        data_path (str, optional): Path to the CSV file
        remove_outliers (bool): Whether to remove outliers
    
    Returns:
        tuple: (X, y, df_processed) - features, target, and processed dataframe
    """
    # Load data
    df = load_data(data_path)
    
    # Remove outliers
    if remove_outliers:
        df = remove_outliers_iqr(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare features
    X, y = prepare_features(df)
    
    return X, y, df

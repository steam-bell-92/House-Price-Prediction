"""
Inference module for the House Price Prediction model.
Provides functions to load the trained model and make predictions.
"""
import numpy as np
import pandas as pd
import joblib
import config


def load_model(model_path=None):
    """
    Load the trained model from disk.
    
    Args:
        model_path (str, optional): Path to the model file. Uses config.MODEL_SAVE_PATH if None.
    
    Returns:
        Trained model
    """
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH
    
    model = joblib.load(model_path)
    return model


def prepare_input_features(area, total_rooms, stories, has_parking, mainroad):
    """
    Prepare input features for prediction.
    
    Args:
        area (float): House area in square feet
        total_rooms (int): Total number of rooms (bedrooms + bathrooms)
        stories (int): Number of stories
        has_parking (int or str): Whether house has parking (1/0 or 'yes'/'no')
        mainroad (int or str): Whether house is on main road (1/0 or 'yes'/'no')
    
    Returns:
        pd.DataFrame: Prepared features
    """
    # Convert string inputs to integers if needed
    if isinstance(has_parking, str):
        has_parking = 1 if has_parking.lower() in ['yes', '1', 'true'] else 0
    if isinstance(mainroad, str):
        mainroad = 1 if mainroad.lower() in ['yes', '1', 'true'] else 0
    
    # Create log transformation for area
    log_area = np.log(area)
    
    # Create feature dictionary
    features = {
        'log_area': log_area,
        'total_rooms': total_rooms,
        'stories': stories,
        'has_parking': has_parking,
        'mainroad': mainroad
    }
    
    # Convert to DataFrame with correct column order
    features_df = pd.DataFrame([features], columns=config.FEATURE_COLUMNS)
    
    return features_df


def make_prediction(area, total_rooms, stories, has_parking, mainroad, 
                   model=None, return_log=False):
    """
    Make a house price prediction based on input features.
    
    This function loads the saved model and scaler (if needed), processes the input,
    and returns the predicted price. It handles single-row inputs as required by web apps.
    
    Args:
        area (float): House area in square feet
        total_rooms (int): Total number of rooms (bedrooms + bathrooms)
        stories (int): Number of stories
        has_parking (int or str): Whether house has parking (1/0 or 'yes'/'no')
        mainroad (int or str): Whether house is on main road (1/0 or 'yes'/'no')
        model (optional): Pre-loaded model. If None, loads from config.MODEL_SAVE_PATH
        return_log (bool): If True, returns log price instead of actual price
    
    Returns:
        float: Predicted house price (or log price if return_log=True)
    
    Example:
        >>> price = make_prediction(3000, 5, 2, 1, 1)
        >>> print(f"Predicted price: ₹{price:,.2f}")
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Prepare input features
    X = prepare_input_features(area, total_rooms, stories, has_parking, mainroad)
    
    # Make prediction (this returns log price)
    log_price = model.predict(X)[0]
    
    # Return log price or actual price
    if return_log:
        return log_price
    else:
        # Convert log price back to actual price
        price = np.exp(log_price)
        return price


def batch_predict(input_data, model=None, return_log=False):
    """
    Make predictions for multiple houses at once.
    
    Args:
        input_data (pd.DataFrame or list of dicts): Input data with columns:
            ['area', 'total_rooms', 'stories', 'has_parking', 'mainroad']
        model (optional): Pre-loaded model. If None, loads from config.MODEL_SAVE_PATH
        return_log (bool): If True, returns log prices instead of actual prices
    
    Returns:
        np.ndarray: Array of predicted prices
    
    Example:
        >>> data = pd.DataFrame({
        ...     'area': [3000, 4000, 5000],
        ...     'total_rooms': [5, 6, 7],
        ...     'stories': [2, 2, 3],
        ...     'has_parking': [1, 1, 0],
        ...     'mainroad': [1, 1, 1]
        ... })
        >>> prices = batch_predict(data)
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Convert list of dicts to DataFrame if needed
    if isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
    
    # Prepare all features
    X_list = []
    for _, row in input_data.iterrows():
        X = prepare_input_features(
            row['area'],
            row['total_rooms'],
            row['stories'],
            row['has_parking'],
            row['mainroad']
        )
        X_list.append(X)
    
    # Combine all features
    X_all = pd.concat(X_list, ignore_index=True)
    
    # Make predictions
    log_prices = model.predict(X_all)
    
    # Return log prices or actual prices
    if return_log:
        return log_prices
    else:
        prices = np.exp(log_prices)
        return prices


def get_model_info(model=None):
    """
    Get information about the trained model.
    
    Args:
        model (optional): Pre-loaded model. If None, loads from config.MODEL_SAVE_PATH
    
    Returns:
        dict: Model information including coefficients and parameters
    """
    if model is None:
        model = load_model()
    
    info = {
        'intercept': float(model.intercept_),
        'coefficients': {
            name: float(coef) 
            for name, coef in zip(config.FEATURE_COLUMNS, model.coef_)
        },
        'alpha': model.alpha,
        'l1_ratio': model.l1_ratio
    }
    
    return info


def main():
    """
    Example usage of the inference module.
    """
    print("\n" + "="*50)
    print("House Price Prediction - Inference Example")
    print("="*50 + "\n")
    
    # Load model once for multiple predictions
    print("Loading trained model...")
    model = load_model()
    print(f"Model loaded from: {config.MODEL_SAVE_PATH}\n")
    
    # Get model info
    print("Model Information:")
    info = get_model_info(model)
    print(f"  Intercept: {info['intercept']:.4f}")
    print(f"  Alpha: {info['alpha']}")
    print(f"  L1 Ratio: {info['l1_ratio']}")
    print("\n  Coefficients:")
    for feature, coef in info['coefficients'].items():
        print(f"    {feature:20s}: {coef:.4f}")
    print()
    
    # Example 1: Single prediction
    print("\nExample 1: Single Prediction")
    print("-" * 50)
    area = 3000
    total_rooms = 5
    stories = 2
    has_parking = 1
    mainroad = 1
    
    print(f"Input:")
    print(f"  Area: {area} sq ft")
    print(f"  Total Rooms: {total_rooms}")
    print(f"  Stories: {stories}")
    print(f"  Has Parking: {has_parking}")
    print(f"  On Main Road: {mainroad}")
    
    price = make_prediction(area, total_rooms, stories, has_parking, mainroad, model=model)
    print(f"\nPredicted Price: ₹{price:,.2f}")
    
    # Example 2: Batch prediction
    print("\n\nExample 2: Batch Prediction")
    print("-" * 50)
    data = pd.DataFrame({
        'area': [2500, 4000, 6000],
        'total_rooms': [4, 6, 8],
        'stories': [1, 2, 3],
        'has_parking': [0, 1, 1],
        'mainroad': [1, 1, 1]
    })
    
    print("Input data:")
    print(data.to_string(index=False))
    
    prices = batch_predict(data, model=model)
    print("\nPredicted Prices:")
    for i, price in enumerate(prices):
        print(f"  House {i+1}: ₹{price:,.2f}")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()

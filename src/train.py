"""
Training script for the House Price Prediction model.
Loads data, trains ElasticNet model, evaluates performance, and saves artifacts.
"""
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import config
import preprocessing


def train_model(X_train, y_train):
    """
    Train a model with configured hyperparameters.
    Uses ElasticNet or LinearRegression based on config.MODEL_TYPE.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained model
    """
    if config.MODEL_TYPE == 'linear':
        model = LinearRegression()
    else:  # elasticnet
        model = ElasticNet(
            alpha=config.ALPHA,
            l1_ratio=config.L1_RATIO,
            random_state=config.RANDOM_STATE
        )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n" + "="*50)
    print("Model Evaluation Metrics")
    print("="*50)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("="*50 + "\n")
    
    return {
        'r2_score': r2,
        'rmse': rmse
    }


def cross_validate_model(model, X, y):
    """
    Perform repeated K-fold cross-validation.
    
    Args:
        model: Model to validate
        X: All features
        y: All targets
    
    Returns:
        dict: Cross-validation results
    """
    print("\n" + "="*50)
    print("Performing Repeated K-Fold Cross-Validation")
    print("="*50)
    
    rkf = RepeatedKFold(
        n_splits=config.N_SPLITS,
        n_repeats=config.N_REPEATS,
        random_state=config.RANDOM_STATE
    )
    
    scores = cross_val_score(model, X, y, cv=rkf, scoring='r2')
    
    avg_r2 = np.mean(scores)
    std_r2 = np.std(scores)
    
    print(f"Average R² Score: {avg_r2:.4f}")
    print(f"Standard Deviation: {std_r2:.4f}")
    print(f"Min R² Score: {np.min(scores):.4f}")
    print(f"Max R² Score: {np.max(scores):.4f}")
    print("="*50 + "\n")
    
    return {
        'avg_r2': avg_r2,
        'std_r2': std_r2,
        'min_r2': np.min(scores),
        'max_r2': np.max(scores)
    }


def save_model(model, save_path=None):
    """
    Save the trained model to disk using joblib.
    
    Args:
        model: Trained model
        save_path (str, optional): Path to save the model. Uses config.MODEL_SAVE_PATH if None.
    """
    if save_path is None:
        save_path = config.MODEL_SAVE_PATH
    
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")


def print_model_coefficients(model, feature_names):
    """
    Print the model coefficients in a readable format.
    
    Args:
        model: Trained model
        feature_names: List of feature names
    """
    print("\n" + "="*50)
    print("Model Coefficients")
    print("="*50)
    print(f"Intercept: {model.intercept_:.4f}")
    print("\nFeature Coefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name:20s}: {coef:.4f}")
    print("="*50 + "\n")


def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*50)
    print("House Price Prediction - Training Pipeline")
    print("="*50 + "\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    X, y, df_processed = preprocessing.preprocess_pipeline()
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {config.TARGET_COLUMN}")
    
    # Step 2: Split data
    print("\nStep 2: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    if config.MODEL_TYPE == 'linear':
        print(f"Model type: LinearRegression")
    else:
        print(f"Model type: ElasticNet (alpha={config.ALPHA}, l1_ratio={config.L1_RATIO})")
    model = train_model(X_train, y_train)
    print("Model training completed!")
    
    # Step 4: Print model coefficients
    print_model_coefficients(model, config.FEATURE_COLUMNS)
    
    # Step 5: Evaluate on test set
    print("Step 4: Evaluating model on test set...")
    test_metrics = evaluate_model(model, X_test, y_test)
    
    # Step 6: Cross-validation
    print("Step 5: Performing cross-validation...")
    cv_metrics = cross_validate_model(model, X, y)
    
    # Step 7: Fit and save scaler (optional, for future use)
    print("Step 6: Saving scaler...")
    scaler = preprocessing.fit_scaler(X_train)
    preprocessing.save_scaler(scaler)
    
    # Step 8: Save model
    print("Step 7: Saving trained model...")
    save_model(model)
    
    print("\n" + "="*50)
    print("Training pipeline completed successfully!")
    print("="*50 + "\n")
    
    # Summary
    print("Summary:")
    if config.MODEL_TYPE == 'linear':
        print(f"  Model: LinearRegression")
    else:
        print(f"  Model: ElasticNet (alpha={config.ALPHA}, l1_ratio={config.L1_RATIO})")
    print(f"  Test R² Score: {test_metrics['r2_score']:.4f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Cross-Validation Avg R²: {cv_metrics['avg_r2']:.4f} ± {cv_metrics['std_r2']:.4f}")
    print(f"  Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"  Scaler saved to: {config.SCALER_SAVE_PATH}")
    print()


if __name__ == "__main__":
    main()

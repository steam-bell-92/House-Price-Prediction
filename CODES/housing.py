import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class HousingPriceModel:
    def __init__(self, model_path='housing_model.pkl'):
        """
        Initialize the model wrapper.
        """
        self.model = LinearRegression()
        self.model_path = model_path
        self.data = None
        self.df_clean = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Features used for the final model
        self.features = ['log_area', 'total_rooms', 'stories', 'has_parking', 'mainroad']

    def load_data(self, path):
        """
        Loads data from CSV.
        """
        self.data = pd.read_csv(path)
        print(f"Data loaded. Shape: {self.data.shape}")

    def preprocess(self, remove_outliers=True):
        """
        Cleaning, Feature Engineering, and Transformation.
        """
        if self.data is None:
            raise ValueError("Data not loaded.")
            
        df = self.data.copy()

        # 1. Feature Engineering
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        
        # 2. Encoding Binary Columns
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                       'prefarea', 'airconditioning']
        
        # Helper to map yes/no to 1/0
        def binary_map(x):
            return 1 if x in ['yes', True] else 0

        for col in binary_cols:
            df[col] = df[col].apply(binary_map)

        # Parking Logic
        df['has_parking'] = df['parking'].apply(lambda x: 1 if x > 0 else 0)

        # 3. Outlier Removal (IQR) - Optional but recommended based on your script
        if remove_outliers:
            cols_to_check = ['price', 'area', 'total_rooms']
            for col in cols_to_check:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            
            print(f"Outliers removed. New Shape: {df.shape}")

        # 4. Log Transformations (Critical for this specific model)
        # We use log1p to avoid log(0) errors, though area/price are usually > 0
        df['log_prices'] = np.log(df['price'])
        df['log_area'] = np.log(df['area'])

        self.df_clean = df
        
        # Prepare X and y
        X = df[self.features]
        y = df['log_prices']
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train(self):
        """
        Trains the Linear Regression model.
        """
        print("Training model...")
        self.model.fit(self.X_train, self.y_train)
        
        # Coefficients
        print("Model learned coefficients:")
        for feat, coef in zip(self.features, self.model.coef_):
            print(f"  {feat}: {coef:.4f}")
        print(f"  Intercept: {self.model.intercept_:.4f}")

    def evaluate(self):
        """
        Evaluates using R2 and Cross Validation.
        """
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        
        # Cross Validation
        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv, scoring='r2')
        
        print(f"\n--- Evaluation ---")
        print(f"Test Set R2 Score: {r2:.4f}")
        print(f"CV Average R2: {np.mean(scores):.4f}")
        return r2

    def predict_price(self, area, total_rooms, stories, has_parking, mainroad):
        """
        Predicts price for a single house given raw inputs.
        Handles the log-transformation logic internally.
        """
        # 1. Transform input (Log Area)
        log_area_val = np.log(area)
        
        # 2. Prepare feature vector
        # Must match order: ['log_area', 'total_rooms', 'stories', 'has_parking', 'mainroad']
        features = np.array([[log_area_val, total_rooms, stories, has_parking, mainroad]])
        
        # 3. Predict (Result is in Log Scale)
        log_price_pred = self.model.predict(features)[0]
        
        # 4. Inverse Transform (Exponential) to get actual currency
        price_pred = np.exp(log_price_pred)
        
        return price_pred

# --- Usage ---

if __name__ == "__main__":
    # Initialize
    house_model = HousingPriceModel()
    
    # Load (Assuming file exists)
    # house_model.load_data('/content/drive/MyDrive/Housing.csv') 
    
    # For demonstration, creating dummy data if file not found
    try:
        house_model.load_data('Housing.csv')
    except:
        print("File not found, skipping execution.")

    # Run Pipeline
    if house_model.data is not None:
        house_model.preprocess(remove_outliers=True)
        house_model.train()
        house_model.evaluate()

        # Prediction Example
        estimated_price = house_model.predict_price(
            area=5000, 
            total_rooms=3, 
            stories=2, 
            has_parking=1, 
            mainroad=1
        )
        print(f"\nEstimated Price for 5000sqft house: ${estimated_price:,.2f}")
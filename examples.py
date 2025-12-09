"""
Example usage of the House Price Prediction ML Pipeline

This script demonstrates how to use the refactored ML pipeline
for training and making predictions.
"""

import sys
sys.path.insert(0, 'src')

from inference import make_prediction, batch_predict, load_model, get_model_info
import pandas as pd

print("\n" + "="*60)
print("House Price Prediction - Usage Examples")
print("="*60)

# Example 1: Load model and get information
print("\n[Example 1] Load model and view information")
print("-" * 60)
model = load_model()
info = get_model_info(model)
print(f"Model Type: {info['model_type']}")
print(f"Intercept: {info['intercept']:.4f}")
print("\nCoefficients:")
for feature, coef in info['coefficients'].items():
    print(f"  {feature:20s}: {coef:.4f}")

# Example 2: Single prediction with numeric inputs
print("\n[Example 2] Single prediction (numeric inputs)")
print("-" * 60)
price = make_prediction(
    area=3000,          # Square feet
    total_rooms=5,      # Total rooms (bedrooms + bathrooms)
    stories=2,          # Number of stories
    has_parking=1,      # 1 = Yes, 0 = No
    mainroad=1          # 1 = Yes, 0 = No
)
print(f"Input: 3000 sq ft, 5 rooms, 2 stories, parking=yes, mainroad=yes")
print(f"Predicted Price: ₹{price:,.2f}")

# Example 3: Single prediction with string inputs
print("\n[Example 3] Single prediction (string inputs)")
print("-" * 60)
price = make_prediction(
    area=5000,
    total_rooms=6,
    stories=2,
    has_parking='yes',  # String input
    mainroad='no'       # String input
)
print(f"Input: 5000 sq ft, 6 rooms, 2 stories, parking='yes', mainroad='no'")
print(f"Predicted Price: ₹{price:,.2f}")

# Example 4: Batch prediction
print("\n[Example 4] Batch prediction")
print("-" * 60)
houses = pd.DataFrame({
    'area': [2500, 4000, 6000, 8000],
    'total_rooms': [4, 6, 8, 9],
    'stories': [1, 2, 3, 3],
    'has_parking': [0, 1, 1, 1],
    'mainroad': [1, 1, 1, 0]
})

print("Input data:")
print(houses.to_string(index=False))

prices = batch_predict(houses, model=model)
print("\nPredicted prices:")
for i, (_, house) in enumerate(houses.iterrows()):
    print(f"  House {i+1} ({house['area']} sq ft, {house['total_rooms']} rooms): "
          f"₹{prices[i]:,.2f}")

# Example 5: Use in a web application (Flask example)
print("\n[Example 5] Flask/Django web app integration")
print("-" * 60)
print("""
# Flask example:
from flask import Flask, request, jsonify
from src.inference import make_prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        price = make_prediction(
            area=float(data['area']),
            total_rooms=int(data['total_rooms']),
            stories=int(data['stories']),
            has_parking=data['has_parking'],  # Can be 1/0 or 'yes'/'no'
            mainroad=data['mainroad']
        )
        return jsonify({
            'success': True,
            'predicted_price': float(price)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
""")

print("\n" + "="*60)
print("For more information, see src/README.md")
print("="*60 + "\n")

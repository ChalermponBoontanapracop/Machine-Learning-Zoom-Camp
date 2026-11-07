# 5.2 Building a Prediction API with Flask

import pickle
import math
import pandas as pd

from flask import Flask, request, jsonify

# Load saved model and preprocessing information
with open("model.bin", "rb") as file:
    artifacts = pickle.load(file)

model = artifacts["model"]
encoder = artifacts["encoder"]
feature_names = artifacts["features"]
selected_sectors = artifacts["sectors"]

# Create Flask application
app = Flask(__name__)


# Check that the API is running
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "PONG"})


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    # Receive JSON input
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    try:
        province = data["province_en"]
        sector = data["sector"]
        users = float(data["number_of_users"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid or missing input"}), 400

    # Validate input values
    valid_provinces = encoder.categories_[0]

    if (
        not isinstance(province, str)
        or province not in valid_provinces
        or sector not in selected_sectors
        or not math.isfinite(users)
        or users < 0
    ):
        return jsonify({"error": "Invalid input values"}), 400

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "province_en": province,
        "sector": sector,
        "number_of_users": users
    }])

    # One-Hot Encoding
    categorical_columns = ["province_en", "sector"]

    encoded_values = encoder.transform(
        input_data[categorical_columns]
    )

    X = pd.DataFrame(
        encoded_values,
        columns=encoder.get_feature_names_out(),
        index=input_data.index
    )

    # Add numerical feature
    X["number_of_users"] = input_data["number_of_users"]

    # Add sector-specific interaction features
    for name in selected_sectors:
        X[f"users_{name}"] = (
            input_data["number_of_users"]
            * (input_data["sector"] == name)
        )

    # Match the feature order used during training
    X = X[feature_names]

    # Generate prediction
    prediction = model.predict(X)[0]

    # Return JSON response
    return jsonify({
        "province_en": province,
        "sector": sector,
        "number_of_users": users,
        "predicted_electricity_kwh": float(prediction)
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)


# ==================================================
# Thailand Electricity Consumption Prediction
# Flask REST API + Web Application
# Historical Analytics + Machine Learning Prediction
# ==================================================

import pickle
import math
from pathlib import Path

import pandas as pd

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory
)

# Import Historical Analytics Blueprint
from historical import historical_bp


# ==================================================
# 1. Project Configuration
# ==================================================

# Get the current project directory
BASE_DIR = Path(__file__).resolve().parent

# Path to the trained Machine Learning model
MODEL_PATH = BASE_DIR / "model.bin"


# ==================================================
# 2. Load Trained Model
# ==================================================

# Load model and preprocessing artifacts
with open(MODEL_PATH, "rb") as file:
    artifacts = pickle.load(file)

# Extract trained model
model = artifacts["model"]

# Extract fitted One-Hot Encoder
encoder = artifacts["encoder"]

# Extract feature names used during training
feature_names = artifacts["features"]

# Extract supported electricity sectors
selected_sectors = artifacts["sectors"]


# ==================================================
# 3. Create Flask Application
# ==================================================

app = Flask(__name__)

# Register the Historical Analytics Blueprint
# This enables GET /historical
app.register_blueprint(historical_bp)


# ==================================================
# 4. Web Application
# ==================================================

@app.route("/", methods=["GET"])
def home():
    """
    Display the Thailand Electricity Analytics
    and Prediction Web Application.
    """

    return send_from_directory(
        BASE_DIR / "web",
        "index.html"
    )


# ==================================================
# 5. API Health Check
# ==================================================

@app.route("/ping", methods=["GET"])
def ping():
    """
    Check whether the Flask API is running.
    """

    return jsonify({
        "message": "PONG"
    })


# ==================================================
# 6. Province and Sector Options
# ==================================================

@app.route("/options", methods=["GET"])
def options():
    """
    Return the provinces and electricity sectors
    supported by the trained Machine Learning model.

    The Web Application uses this endpoint to
    populate the dropdown menus.
    """

    # Retrieve provinces from the fitted encoder
    provinces = sorted(
        str(province)
        for province in encoder.categories_[0]
    )

    # Retrieve electricity sectors
    sectors = list(selected_sectors)

    return jsonify({
        "provinces": provinces,
        "sectors": sectors
    })


# ==================================================
# 7. Machine Learning Prediction API
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict annual electricity consumption
    using the trained Random Forest model.

    Expected JSON Input:
    {
        "province_en": "Chiang Mai",
        "sector": "residential",
        "number_of_users": 500000
    }
    """

    # ----------------------------------------------
    # Step 1: Receive JSON Input
    # ----------------------------------------------

    data = request.get_json(silent=True)

    if not isinstance(data, dict):

        return jsonify({
            "error": "A JSON object is required"
        }), 400

    # ----------------------------------------------
    # Step 2: Extract Input Features
    # ----------------------------------------------

    try:

        province = data["province_en"]

        sector = data["sector"]

        users = float(
            data["number_of_users"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError
    ):

        return jsonify({
            "error": "Invalid or missing input"
        }), 400

    # ----------------------------------------------
    # Step 3: Validate Input Values
    # ----------------------------------------------

    valid_provinces = encoder.categories_[0]

    if (
        not isinstance(province, str)
        or province not in valid_provinces
        or not isinstance(sector, str)
        or sector not in selected_sectors
        or not math.isfinite(users)
        or users < 0
    ):

        return jsonify({
            "error": "Invalid input values"
        }), 400

    # ----------------------------------------------
    # Step 4: Create Input DataFrame
    # ----------------------------------------------

    input_data = pd.DataFrame([{
        "province_en": province,
        "sector": sector,
        "number_of_users": users
    }])

    # ----------------------------------------------
    # Step 5: One-Hot Encoding
    # ----------------------------------------------

    # Categorical features
    categorical_columns = [
        "province_en",
        "sector"
    ]

    # Apply fitted One-Hot Encoder
    encoded_values = encoder.transform(
        input_data[categorical_columns]
    )

    # Convert encoded values into a DataFrame
    X = pd.DataFrame(
        encoded_values,
        columns=encoder.get_feature_names_out(),
        index=input_data.index
    )

    # ----------------------------------------------
    # Step 6: Numerical Feature
    # ----------------------------------------------

    # Add the number of electricity users
    X["number_of_users"] = (
        input_data["number_of_users"]
    )

    # ----------------------------------------------
    # Step 7: Sector Interaction Features
    # ----------------------------------------------

    # Create sector-specific number-of-users features
    for name in selected_sectors:

        X[f"users_{name}"] = (
            input_data["number_of_users"]
            * (input_data["sector"] == name)
        )

    # ----------------------------------------------
    # Step 8: Match Training Feature Order
    # ----------------------------------------------

    X = X[feature_names]

    # ----------------------------------------------
    # Step 9: Generate Prediction
    # ----------------------------------------------

    prediction = model.predict(X)[0]

    # ----------------------------------------------
    # Step 10: Return JSON Response
    # ----------------------------------------------

    return jsonify({
        "province_en": province,
        "sector": sector,
        "number_of_users": users,
        "predicted_electricity_kwh": float(prediction)
    })


# ==================================================
# 8. Run Flask Application
# ==================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

"""Simple HTTP smoke tests for the Thailand electricity prediction API.

Start the API first (locally or in Docker), then run:
    python -m pipenv run python test.py

Optional: set API_URL to test another address.
"""

import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


base_url = os.environ.get("API_URL", "http://127.0.0.1:5000").rstrip("/")


def send_request(path, method="GET", data=None):
    """Send a request and return the status code and JSON response."""
    body = json.dumps(data).encode("utf-8") if data is not None else None
    request = Request(
        base_url + path,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method=method,
    )

    try:
        with urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


try:
    # 1. Check whether the API is running.
    status, response = send_request("/ping")
    assert status == 200, f"Ping returned status {status}"
    assert response.get("message") == "PONG", f"Unexpected ping response: {response}"
    print("PASS: /ping")

    # 2. Send the same example used in the Colab and Docker tests.
    test_input = {
        "province_en": "Chiang Mai",
        "sector": "residential",
        "number_of_users": 500000,
    }
    status, response = send_request("/predict", method="POST", data=test_input)
    assert status == 200, f"Prediction failed: {response}"

    predicted_kwh = response.get("predicted_electricity_kwh")
    assert isinstance(predicted_kwh, (int, float)), "Prediction must be numeric"
    assert math.isfinite(predicted_kwh), "Prediction must be a finite number"
    assert response.get("province_en") == test_input["province_en"]
    assert response.get("sector") == test_input["sector"]
    print("PASS: /predict")
    print(f"Predicted annual electricity consumption: {predicted_kwh:,.2f} kWh")

    # 3. Check that the API rejects an invalid sector.
    invalid_input = test_input.copy()
    invalid_input["sector"] = "not_a_real_sector"
    status, response = send_request("/predict", method="POST", data=invalid_input)
    assert status == 400, f"Expected HTTP 400 for invalid sector, got {status}"
    print("PASS: invalid input rejected")

    print("\nAll API tests passed.")

except (URLError, ConnectionError) as error:
    print(f"Cannot connect to API at {base_url}: {error}")
    raise SystemExit(1)
except (AssertionError, ValueError, KeyError) as error:
    print(f"API test failed: {error}")
    raise SystemExit(1)

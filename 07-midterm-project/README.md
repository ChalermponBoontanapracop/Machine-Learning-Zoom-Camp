# Thailand Electricity Consumption Prediction Across Five Major Sectors

A machine learning project that predicts **annual electricity consumption (kWh)** for a given province and electricity-user sector in Thailand. The project includes data preparation, exploratory analysis, regression model development, evaluation, and a Flask prediction API packaged with Docker.

## Project overview

The analysis uses 12 annual electricity-consumer and electricity-consumption Excel files (six for users and six for kWh), plus a province-name mapping file, covering **2018–2023** and **77 provinces**. The original data includes 12 electricity sectors; the modeling scope focuses on five major sectors:

- `residential`
- `small_business`
- `medium_business`
- `large_business`
- `specific_business`

Each modeling observation represents a **province × year × sector** combination. The target is the **total annual electricity consumption in kWh** for that combination, not consumption per household or per user.

### Inputs and prediction target

| Field | Description |
| --- | --- |
| `province_en` | Province name in English |
| `sector` | One of the five selected electricity sectors |
| `number_of_users` | Number of electricity users in the province and sector |
| `electricity_kwh` | Annual electricity consumption; prediction target |

Province and sector are one-hot encoded. The model also uses the number of users and sector-specific user interaction features. The saved encoder and feature order are reused during inference.

## Key findings from the executed notebook

The following results are taken from the **saved cell outputs of the Colab analysis notebook**, rather than estimated values. They describe the dataset and models used in that notebook run.

### Sector coverage (2018–2023)

The five selected sectors account for **96.34% of recorded electricity consumption** across the 12 original sectors (approximately **1,104.25 billion kWh** of **1,146.17 billion kWh**). The excluded categories account for approximately **3.66%**. The selected modeling table contains **2,310 observations** (77 provinces × 6 years × 5 sectors).

| Selected sector | Share of all recorded sector consumption |
| --- | ---: |
| Large Business | 38.65% |
| Residential | 27.31% |
| Medium Business | 16.10% |
| Small Business | 11.33% |
| Specific Business | 2.95% |
| **Selected five sectors** | **96.34%** |

These percentages are measured against all **recorded** sectors over the full six-year period, not solely within the five selected sectors. Some province-level or historical measurements are incomplete; coverage should not be interpreted as a guarantee of complete electricity-demand accounting.

### Model comparison — 2022 validation

| Model | MAE (million kWh) | RMSE (million kWh) | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 138.03 | 271.48 | 0.9467 |
| Ridge Regression (tuned) | 137.86 | 271.47 | 0.9467 |
| Random Forest (baseline) | 43.92 | 141.62 | 0.9855 |
| **Random Forest (tuned)** | **38.42** | **113.16** | **0.9907** |

The tuned Random Forest achieved the lowest **validation RMSE** in the comparison. The selected search result used `n_estimators=300`, `max_depth=40`, `min_samples_split=2`, `min_samples_leaf=1`, and `max_features="sqrt"`.

### Final evaluation — 2023

| Metric | Tuned Random Forest, 2023 test |
| --- | ---: |
| MAE | **53.67 million kWh** |
| RMSE | **143.37 million kWh** |
| R² | **0.9859** |

Test error was higher than validation error, although the model retained a high R². The test set includes **385 province–sector observations**. The notebook's 2023 results should be considered with the limitation that this year may have been examined during earlier project iterations; it may not constitute a fully untouched holdout.

### Model interpretation

Permutation importance on the 2022 validation data highlighted `number_of_users`, `users_large_business`, `province_en_Bangkok`, and `sector_large_business` among the most influential **encoded features**. Because feature groups overlap or correlate, these scores describe the trained model's predictive dependence, **not causal effects** or unique contributions of the original variables.

## Data preparation and modeling

The notebook covers loading and integrating yearly spreadsheets, province-name mapping, data quality checks, wide-to-long transformation, sector selection, exploratory data analysis, and historical anomaly investigation. Potential historical anomalies are screened using an exploratory **±50% year-over-year change** threshold and handled with a **historical median adjustment applied only to training targets**. In the saved notebook run, **25 training observations were flagged and adjusted**. Flags are screening indicators, not confirmed errors; the original source files remain unchanged.

Data is separated chronologically:

| Partition | Years | Use |
| --- | --- | --- |
| Training | 2018–2021 | Fit models and preprocessing |
| Validation | 2022 | Evaluate tuning choices and select a model |
| Test | 2023 | Report final performance |

The notebook evaluates four candidates:

1. Linear Regression (baseline)
2. Ridge Regression (tuned regularization)
3. Random Forest Regression (baseline)
4. Random Forest Regression (hyperparameter tuned)

The models are assessed using **MAE, RMSE, and R²**. The model is selected primarily on validation RMSE. The project also examines predictions and errors by sector and province and investigates feature importance. The performance figures above are transcribed from the executed notebook outputs. A successful API response demonstrates serving functionality, not model accuracy.

## Repository structure

```text
thailand-electricity-prediction/
├── README.md
├── notebook.ipynb          # Data preparation, analysis, training and evaluation
├── predict.py              # Flask API
├── model.bin               # Serialized fitted model, encoder and feature metadata
├── Pipfile                 # Python dependencies
├── Pipfile.lock            # Locked dependency versions
├── Dockerfile              # Docker image definition
└── .dockerignore           # Docker build exclusions
```

The `notebook.ipynb` file can be added from the completed analysis notebook. `model.bin` must be produced by the training notebook and placed next to `predict.py`. Do not load untrusted pickle files.

## Run locally with Pipenv

Requirements: Python and Pipenv, with a version of scikit-learn compatible with the saved model. The saved model used in local testing was created with **scikit-learn 1.6.1**; use that version when loading the existing `model.bin`.

Open a terminal in the project directory and install the locked dependencies:

```powershell
python -m pip install pipenv
python -m pipenv sync
```

If you are setting up dependencies from scratch instead of an existing lockfile:

```powershell
python -m pipenv install flask pandas numpy "scikit-learn==1.6.1" waitress
```

Run the API locally:

```powershell
python -m pipenv run python predict.py
```

By default the Flask development server listens at `http://127.0.0.1:5000`. For local use with Waitress:

```powershell
python -m pipenv run waitress-serve --host=127.0.0.1 --port=5000 predict:app
```

## Run with Docker

Requirements: Docker Desktop with a running Docker Engine. The `Dockerfile` installs the locked dependencies and starts the API with Waitress.

Build the image:

```powershell
docker build -t thailand-electricity-api .
```

Run the container:

```powershell
docker run --rm -p 5000:5000 thailand-electricity-api
```

The API can then be reached from the host computer at `http://127.0.0.1:5000`. Stop the container with **Ctrl+C**.

## API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/ping` | Health check; returns `{"message":"PONG"}` |
| `POST` | `/predict` | Predicts annual electricity consumption from a JSON input |

### Example prediction request

```json
{
  "province_en": "Chiang Mai",
  "sector": "residential",
  "number_of_users": 500000
}
```

Example response from the tested saved model:

```json
{
  "province_en": "Chiang Mai",
  "sector": "residential",
  "number_of_users": 500000.0,
  "predicted_electricity_kwh": 1240984762.1780288
}
```

This is approximately **1.241 billion kWh per year**. It is one example prediction, not a forecast for an entire province across all sectors.

### Test in Windows PowerShell

Leave the API running in one terminal. Open another terminal:

```powershell
# Health check
Invoke-RestMethod -Uri "http://127.0.0.1:5000/ping" -Method GET

# Prediction request
$inputData = @{
    province_en = "Chiang Mai"
    sector = "residential"
    number_of_users = 500000
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $inputData
```

The prediction was checked successfully using the Flask test client, the local Windows server, and the Docker container, with matching output for this example request.

## Limitations

- The project uses only **five selected sectors**, so its predictions do not represent total electricity use across all 12 original sectors.
- Annual observations provide limited temporal detail; the model does not forecast monthly demand or account explicitly for temperature, weather, tariffs, and other external factors.
- Predictions depend on the quality and representativeness of the provided `number_of_users` value and the historical data.
- Historical anomaly flags are not proof of data-entry errors. Median adjustment is a modeling assumption and should be reviewed against source records.
- For a completely independent test, no decisions should be made using 2023 results. If that year was consulted during earlier experimentation, state this limitation in the evaluation report.
- The API currently runs locally or inside Docker; running a Docker container on a personal computer does **not** make the service publicly accessible on the internet.

## Next steps

Add reproducible `train.py` and automated `test.py` scripts, document the underlying data sources with exact citations and licenses, and consider cloud deployment after checking API validation, model-version compatibility, and security settings.

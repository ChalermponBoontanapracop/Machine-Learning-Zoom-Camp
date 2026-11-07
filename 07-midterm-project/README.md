# Thailand Electricity Consumption Prediction Across Five Major Sectors

A machine learning project that predicts **annual electricity consumption (kWh)** for a given province and electricity-user sector in Thailand.

The project covers data preparation, exploratory data analysis (EDA), regression modeling, model evaluation, and deployment through a Flask REST API packaged with Docker.

The final selected algorithm is a **Tuned Random Forest Regressor**, deployed locally, through Docker, and on Render Cloud.

**Live Prediction API:** https://thailand-electricity-api.onrender.com

**GitHub Project:** https://github.com/ChalermponBoontanapracop/Machine-Learning-Zoom-Camp/tree/main/07-midterm-project

---

## 1. Project Overview

Electricity consumption varies across provinces and economic sectors due to differences in population, business activity, and the number of electricity users.

This project investigates the relationship between electricity consumption and the number of electricity users across five major sectors in Thailand.

The objective is to develop a machine learning model capable of estimating annual electricity consumption using:

- Province
- Electricity-user sector
- Number of electricity users

The predictions may support exploratory electricity-demand analysis and comparisons across provinces and sectors. They should not be treated as official electricity-demand forecasts.

### Selected Electricity Sectors

The original dataset includes 12 electricity-user sectors. This project focuses on five major sectors:

1. Residential
2. Small Business
3. Medium Business
4. Large Business
5. Specific Business

These five sectors account for approximately **96.34% of recorded electricity consumption** across the 12 original sectors during 2018–2023.

The prediction target is the **total annual electricity consumption in kWh** for a province and sector, rather than consumption per household or individual user.

---

## 2. Dataset Description

### Data Coverage

| Description | Details |
|---|---|
| Country | Thailand |
| Period | 2018–2023 |
| Geographic coverage | 77 provinces |
| Original sectors | 12 |
| Selected sectors | 5 |
| Target variable | Annual electricity consumption (kWh) |
| Main numerical feature | Number of electricity users |
| Modeling observations | 2,310 |

Each observation represents one combination of:

**Province × Year × Electricity Sector**

The original data consists of:

- Six annual Excel files containing electricity-consumer counts
- Six annual Excel files containing electricity consumption in kWh
- One province-name mapping file

The annual spreadsheets are combined into a structured dataset for machine learning.

### Data Sources and Availability

The dataset was obtained from the **Ministry of Energy, Thailand — OPS MOEN Data Catalog**, under the dataset titled **Electricity Consumption (การใช้ไฟฟ้า)**.

**Official Dataset:** https://gdcatalog.energy.go.th/dataset/dataset_41_07

The project uses annual provincial electricity statistics from the published electricity data, including information originating from the Provincial Electricity Authority (PEA) and Metropolitan Electricity Authority (MEA).

Two categories of data were used:

1. **Number of Electricity Users (จำนวนผู้ใช้ไฟฟ้ารายจังหวัดรายสาขา):** Annual electricity-user counts by province and sector.
2. **Electricity Consumption (ปริมาณการใช้ไฟฟ้ารายจังหวัดรายสาขา):** Annual electricity consumption by province and sector, measured in kWh.

The analysis covers **2018–2023 (B.E. 2561–2566)**, using six annual Excel files for each data category, totaling 12 annual files.

A separate province-name mapping file is used to standardize province names in English.

The original Excel files are combined, cleaned, transformed, and prepared for regression modeling. The resulting dataset contains **2,310 province–year–sector observations** across the five selected sectors.

The official data catalog provides the public source reference. The project's `raw data` directory contains the files used for the analysis and model training. Users should consult the original catalog for applicable data reuse conditions.

### Selected Sectors and Electricity Consumption

| Sector | Share of total recorded electricity consumption |
|---|---:|
| Large Business | 38.65% |
| Residential | 27.31% |
| Medium Business | 16.10% |
| Small Business | 11.33% |
| Specific Business | 2.95% |
| **Total selected sectors** | **96.34%** |

The percentages are calculated against recorded electricity consumption across all 12 original sectors during 2018–2023.

The five selected sectors represent approximately **1,104.25 billion kWh** of **1,146.17 billion kWh** recorded across all sectors.

Historical data may contain incomplete observations, and these totals should not be interpreted as complete national electricity-demand accounting.

---

## 3. Features and Target Variable

The machine learning model uses the following input and target variables:

| Variable | Type | Description |
|---|---|---|
| `province_en` | Categorical | Province name in English |
| `sector` | Categorical | Electricity-user sector |
| `number_of_users` | Numerical | Number of electricity users |
| `electricity_kwh` | Numerical | Annual electricity consumption (target) |

### Feature Engineering

Categorical features are transformed using **One-Hot Encoding**.

The model uses:

- Encoded province features
- Encoded sector features
- Number of electricity users
- Sector-specific user interaction features

The interaction features help the model capture differences in electricity consumption associated with user counts across sectors.

The saved encoder and feature order are reused during prediction to maintain consistency between training and inference.

---

## 4. Data Preparation and Exploratory Data Analysis

The notebook documents the following preparation steps:

1. Load annual Excel datasets.
2. Standardize province names.
3. Combine annual electricity-consumer and consumption data.
4. Transform sector columns into structured observations.
5. Select the five major electricity sectors.
6. Investigate missing values and data consistency.
7. Analyze electricity consumption distributions.
8. Explore relationships between electricity users and consumption.
9. Investigate historical anomalies.
10. Prepare training, validation, and test datasets.

### Historical Anomaly Detection

The analysis screens potential historical anomalies using a **±50% year-over-year change threshold**.

This threshold helps identify unusual changes in electricity consumption that require further investigation.

An anomaly flag does not necessarily mean that the original observation is incorrect.

### Historical Median Adjustment

The modeling approach applies a historical median adjustment to selected anomalous training observations.

During the saved analysis:

- Potential anomalies identified: 25
- Training observations adjusted: 25

The adjustment is applied to training targets only.

The original source data remains unchanged.

This approach is a modeling assumption rather than a verified correction of errors in the original statistics.

---

## 5. Training, Validation, and Test Split

The dataset is divided chronologically to evaluate how well the model performs on later observations.

| Dataset | Years | Purpose |
|---|---|---|
| Training | 2018–2021 | Train models |
| Validation | 2022 | Compare models and select hyperparameters |
| Test | 2023 | Evaluate final model performance |

The split avoids randomly mixing earlier and later annual observations.

This is especially relevant when the intended application involves estimating electricity consumption for later periods.

### Dataset Size

| Dataset | Observations |
|---|---:|
| Training | 1,540 |
| Validation | 385 |
| Test | 385 |
| **Total** | **2,310** |

The 2023 test data is not used by `train.py` to fit or select the model.

However, the original notebook results should be interpreted with the limitation that 2023 may have been examined during earlier project iterations.

---

## 6. Machine Learning Models

The project compares three regression algorithms across four model configurations.

### Models Evaluated

1. Linear Regression
2. Ridge Regression with regularization tuning
3. Random Forest Regression (baseline)
4. Random Forest Regression with hyperparameter tuning

### Evaluation Metrics

Three metrics are used:

**Mean Absolute Error (MAE)**

Measures the average absolute difference between predicted and actual electricity consumption.

**Root Mean Squared Error (RMSE)**

Measures prediction error while placing greater emphasis on larger errors.

**R² Score**

Measures how much variation in the target is explained by the model on the evaluated dataset.

RMSE is used as the primary metric for model selection.

---

## 7. Model Comparison

### Validation Results — 2022

The following comparison results are from the saved outputs of the analysis notebook.

| Model | MAE (million kWh) | RMSE (million kWh) | R² |
|---|---:|---:|---:|
| Linear Regression | 138.03 | 271.48 | 0.9467 |
| Ridge Regression (tuned) | 137.86 | 271.47 | 0.9467 |
| Random Forest (baseline) | 43.92 | 141.62 | 0.9855 |
| **Random Forest (tuned)** | **38.42** | **113.16** | **0.9907** |

The Tuned Random Forest achieved the lowest validation RMSE among the evaluated models.

### Selected Hyperparameters

```python
{
    "n_estimators": 300,
    "max_depth": 40,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt"
}
```

### Reproducible Training Script Results

The independently executed `train.py` produced the following 2022 validation results:

| Metric | Result |
|---|---:|
| MAE | 38.15 million kWh |
| RMSE | 113.16 million kWh |
| R² | 0.9907 |

The training script reported:

- Training observations: 1,540
- Validation observations: 385
- Historical median adjustments: 25

### Evaluation Consistency — Notebook vs. Training Script

To assess reproducibility, the selected Tuned Random Forest was evaluated using both the original analysis notebook (`notebook.ipynb`) and the standalone training script (`train.py`).

Both workflows used the 2022 validation dataset and produced closely aligned evaluation results.

| Evaluation Metric | Analysis Notebook | Training Script (`train.py`) | Difference |
|---|---:|---:|---:|
| MAE (million kWh) | 38.42 | 38.15 | 0.27 |
| RMSE (million kWh) | 113.16 | 113.16 | ~0.00 |
| R² | 0.9907 | 0.9907 | ~0.0000 |

**Evaluation Notes**

The two workflows produced approximately the same RMSE and R², indicating similar overall predictive performance on the 2022 validation dataset.

A small difference was observed in MAE, with the training script producing approximately **0.27 million kWh (0.7%)** lower error than the original notebook.

The exact cause of this difference has not been verified. Therefore, the results are reported separately rather than treated as identical.

**Reproducibility Verification**

The `train.py` script was successfully executed from a fresh GitHub clone. The reproducibility test confirmed that the project could install its dependencies, load the datasets, train the selected model, generate `model.bin`, and serve predictions through the Flask API.

The repeated training execution reproduced the training script's reported validation metrics.

Overall, the evaluation results support the selection of the Tuned Random Forest while transparently documenting the small difference between the notebook and standalone training workflows.

---

## 8. Final Model Evaluation

### Test Results — 2023

The following results were obtained from the saved notebook evaluation of the Tuned Random Forest.

| Metric | 2023 Test Result |
|---|---:|
| MAE | 53.67 million kWh |
| RMSE | 143.37 million kWh |
| R² | 0.9859 |

The test RMSE is higher than the validation RMSE.

Nevertheless, the model retains a high R² on the 2023 observations.

These results are historical notebook outputs and are not a newly executed independent test of the latest `train.py` artifact.

### Model Interpretation

Permutation importance was examined using the validation dataset.

Influential encoded features included:

- `number_of_users`
- `users_large_business`
- `province_en_Bangkok`
- `sector_large_business`

These results suggest that electricity-user counts and sector-related features are important to the model's predictions.

However, feature importance does not establish causation, particularly where encoded features and interaction variables are correlated.

---

## 9. Project Structure

The project is stored in the `07-midterm-project` directory.

```text
07-midterm-project/
├── raw data/                 # Original electricity datasets
├── notebook.ipynb            # EDA, modeling and evaluation
├── train.py                  # Reproducible model training
├── predict.py                # Flask prediction API
├── test.py                   # API test script
├── model.bin                 # Saved model and preprocessing artifacts
├── Pipfile                   # Python dependencies
├── Pipfile.lock              # Locked dependency versions
├── Dockerfile                # Docker configuration
├── .dockerignore             # Docker build exclusions
├── .gitignore                # Git exclusions
└── README.md                 # Project documentation
```

The notebook contains the exploratory analysis and model development.

The `train.py` script provides a reproducible training workflow.

The `predict.py` application loads `model.bin` and serves predictions through a Flask API.

The `test.py` script checks the API endpoints and input validation.

---

## 10. Environment Setup

### Requirements

- Python 3.11
- Pipenv
- Docker Desktop (for container deployment)

The project uses the following main Python libraries:

- pandas
- numpy
- scikit-learn
- openpyxl
- Flask
- Waitress

The original saved deployment model was tested with scikit-learn 1.6.1. When loading a saved model, use compatible dependency versions from the project lockfile.

### Install Pipenv

```powershell
python -m pip install pipenv
```

### Install Project Dependencies

Open PowerShell in the project directory.

```powershell
python -m pipenv sync
```

This command installs the dependencies specified in `Pipfile.lock`.

The updated lockfile includes `openpyxl`, which is required for reading the original Excel datasets.

For reproducibility, the committed `Pipfile` and `Pipfile.lock` should be used instead of independently installing arbitrary dependency versions.

---

## 11. Train the Model

The `train.py` script reproduces the final selected model training workflow without requiring manual execution of the notebook.

### Training Workflow

1. Load the annual electricity datasets.
2. Standardize and combine the data.
3. Select the five major sectors.
4. Apply historical median adjustment to training observations.
5. Prepare the numerical and categorical features.
6. Encode province and sector information.
7. Train the selected Tuned Random Forest.
8. Evaluate predictions on 2022 validation data.
9. Save the fitted model and preprocessing metadata to `model.bin`.

### Data Requirements

The original Excel datasets and province-name mapping must be accessible to the training script.

The source data used for this project is available through the official Ministry of Energy data catalog:

https://gdcatalog.energy.go.th/dataset/dataset_41_07

The repository also contains the `raw data` directory used for training.

The training script should be run with its expected input filenames and directory structure.

### Run Training

Open PowerShell in the project directory.

```powershell
python -m pipenv run python train.py
```

The script generates:

```text
model.bin
```

The artifact contains the fitted model and the metadata required by the prediction application.

**Important:** Running `train.py` overwrites the existing `model.bin`. Back up the artifact before retraining if the original version must be preserved.

### Latest Verified Training Execution

The script successfully completed training with:

```text
Potential anomalies: 25
Median-adjusted rows: 25
Training observations: 1540
Validation observations: 385
Test observations (not used): 385

2022 validation MAE: 38146902.46
2022 validation RMSE: 113160100.36
2022 validation R2: 0.9907

Saved: model.bin
```

The test set from 2023 was not used to fit or select the model in this training run.

---

## 12. Run the Flask API Locally

The prediction service is implemented in `predict.py`.

The API loads the saved model and preprocessing artifacts from `model.bin`.

### Start the Server

```powershell
python -m pipenv run python predict.py
```

The Flask development server runs at:

```text
http://127.0.0.1:5000
```

The development server is intended for local testing.

### Run with Waitress

For a production-style local WSGI server:

```powershell
python -m pipenv run waitress-serve --host=127.0.0.1 --port=5000 predict:app
```

---

## 13. API Endpoints

The API provides two endpoints.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/ping` | Health check |
| POST | `/predict` | Predict annual electricity consumption |

### Health Check

Request:

```http
GET /ping
```

Expected response:

```json
{
  "message": "PONG"
}
```

### Prediction Endpoint

Request:

```http
POST /predict
Content-Type: application/json
```

Example JSON body:

```json
{
  "province_en": "Chiang Mai",
  "sector": "residential",
  "number_of_users": 500000
}
```

Example response:

```json
{
  "province_en": "Chiang Mai",
  "sector": "residential",
  "number_of_users": 500000.0,
  "predicted_electricity_kwh": 1240984762.1780288
}
```

This represents approximately **1.241 billion kWh per year** for the specified province, sector, and number of users.

It is not a prediction of the province's total electricity consumption across all sectors.

---

## 14. Test the Prediction API

Start the Flask application in one terminal.

```powershell
python -m pipenv run python predict.py
```

Open another PowerShell terminal in the same project directory.

Run:

```powershell
python -m pipenv run python test.py
```

### Verified Test Results

The API was tested after regenerating `model.bin` using `train.py`.

```text
PASS: /ping
PASS: /predict
Predicted annual electricity consumption: 1,240,984,762.18 kWh
PASS: invalid input rejected

All API tests passed.
```

The tests confirm that:

- The Flask server responds to health checks.
- The prediction endpoint accepts valid input.
- The model returns a numerical prediction.
- Invalid input is rejected.

Successful API tests demonstrate serving functionality, but do not independently establish predictive accuracy.

---

## 15. Docker Deployment

The project includes a Dockerfile to package the prediction service and its dependencies.

Docker provides a consistent environment for running the application on other machines.

### Requirements

- Docker Desktop installed
- Docker Engine running

### Build Docker Image

Run the following command from the project directory:

```powershell
docker build -t thailand-electricity-api .
```

A fresh build without Docker cache can also be performed:

```powershell
docker build --no-cache -t thailand-electricity-api .
```

The no-cache build was successfully completed using the updated `model.bin` and project dependencies.

### Run Docker Container

```powershell
docker run --rm -p 5000:5000 thailand-electricity-api
```

The prediction API will be accessible at:

```text
http://127.0.0.1:5000
```

The Docker configuration uses Waitress to serve the Flask application.

If port 5000 is already occupied on the host machine, another host port can be used:

```powershell
docker run --rm -p 5001:5000 thailand-electricity-api
```

In this configuration, the API is accessible at:

```text
http://127.0.0.1:5001
```

### Docker Deployment Verification

The Docker image was successfully rebuilt using the updated model artifact and latest dependencies.

The container was tested using port 5001 on the host machine, mapped to port 5000 inside Docker.

**Verification Results**

| Test | Endpoint | Result |
|---|---|---|
| Container startup | Waitress server | Passed |
| Health Check | `GET /ping` | Passed |
| Prediction | `POST /predict` | Passed |

### Verified Docker Prediction

The prediction endpoint was tested with the following input:

```json
{
  "province_en": "Chiang Mai",
  "sector": "residential",
  "number_of_users": 500000
}
```

**Prediction Result:** 1,240,984,762.18 kWh per year.

The Docker API returned the same prediction as the locally tested Flask application.

This confirms that the updated model artifact can be loaded and used successfully through the containerized prediction service.

### Stop the Container

Press `Ctrl+C` in the terminal running the container.

---

## 16. Model Deployment Architecture

```text
Original Electricity Datasets
             |
             v
     Data Preparation
             |
             v
     Feature Engineering
             |
             v
    Random Forest Training
          train.py
             |
             v
          model.bin
             |
             v
      Flask Application
         predict.py
             |
             v
       REST API Endpoint
          /predict
             |
             v
     Electricity Prediction
             |
             v
        JSON Response
```

The training and prediction workflows are separated so that the API can load the fitted model without retraining it for every request.

---

## 17. Reproducibility

The project supports reproducibility through:

- A documented analysis notebook
- A separate training script
- Saved dependency versions
- A serialized model artifact
- A prediction application
- Automated API tests
- Docker configuration
- Setup and execution instructions

### Reproduction Steps

1. Clone the GitHub repository.
2. Open `07-midterm-project`.
3. Ensure the required source datasets are available.
4. Install dependencies with Pipenv.
5. Run `train.py` to generate `model.bin`.
6. Start the prediction API.
7. Run `test.py`.
8. Build and run the Docker image.
9. Test the containerized API.

### Clone Repository

```powershell
git clone https://github.com/ChalermponBoontanapracop/Machine-Learning-Zoom-Camp.git

cd Machine-Learning-Zoom-Camp\07-midterm-project
```

### Install Dependencies

```powershell
python -m pipenv sync
```

### Train Model

```powershell
python -m pipenv run python train.py
```

### Start API

```powershell
python -m pipenv run python predict.py
```

### Run Tests

In a second terminal:

```powershell
python -m pipenv run python test.py
```

### Fresh Clone Reproducibility Verification

A fresh-clone reproducibility test was successfully completed using the project files downloaded directly from GitHub.

The repository was cloned into a separate directory to verify that the machine learning workflow could be reproduced without relying on the original project folder.

**Verification Results**

| Verification Step | Result |
|---|---|
| Clone GitHub Repository | Passed |
| Verify Required Project Files | Passed |
| Install Dependencies from `Pipfile.lock` | Passed |
| Load Electricity Datasets | Passed |
| Execute `train.py` | Passed |
| Generate `model.bin` | Passed |
| Start Flask Prediction API | Passed |
| API Health Check | Passed |
| API Prediction Test | Passed |
| Invalid Input Validation | Passed |

### Reproduced Validation Metrics — 2022

| Metric | Result |
|---|---:|
| MAE | 38.15 million kWh |
| RMSE | 113.16 million kWh |
| R² | 0.9907 |
| Training observations | 1,540 |
| Validation observations | 385 |
| Historical median adjustments | 25 |

The fresh-clone training execution reproduced the validation results obtained from the standalone training script.

The Flask API successfully loaded the newly generated model artifact and returned the expected prediction for the test request.

These results confirm that the project's training and local prediction workflows can be reproduced using the files and instructions provided in the GitHub repository.

### Verified Git Commit

```text
51322265deb3d4ac61ec4d6ba5581899a93c18e8
```

This commit identifies the repository version used during the successful fresh-clone test. Subsequent README updates and cloud deployment documentation may belong to later commits.

---

## 18. Cloud Deployment — Render

The final machine learning model was successfully deployed to **Render Cloud** using Docker, Flask, and Waitress.

The deployment provides a public HTTPS API that allows external users and applications to request annual electricity consumption predictions without installing Python, Docker, or the model locally.

### Cloud Service Information

| Configuration | Details |
|---|---|
| Cloud Platform | Render |
| Service Name | `thailand-electricity-api` |
| Deployment Source | GitHub Repository |
| Branch | `main` |
| Root Directory | `07-midterm-project` |
| Runtime | Docker |
| Web Framework | Flask |
| WSGI Server | Waitress |
| Health Check Path | `/ping` |
| Instance Type | Free |
| Deployment Status | Successfully deployed and tested |

**Public API URL:** https://thailand-electricity-api.onrender.com

### Cloud Deployment Architecture

```text
GitHub Repository
       |
       v
Render Cloud Deployment
       |
       v
Docker Image Build
       |
       v
Flask API + Waitress
       |
       v
Tuned Random Forest Model
       |
       v
Public HTTPS API
       |
       v
Electricity Consumption Prediction
```

Render builds the Docker image from the GitHub repository and runs the Flask prediction service with the saved `model.bin` artifact.

### Render Configuration

The cloud deployment uses the following configuration:

| Setting | Value |
|---|---|
| Language | Docker |
| Branch | `main` |
| Root Directory | `07-midterm-project` |
| Docker Build Context | `.` relative to the configured root directory |
| Dockerfile | `Dockerfile` within the project directory |
| Health Check | `/ping` |
| Auto-Deploy | On Commit |
| Compute | Free instance |

The existing project Dockerfile is used for deployment. No separate Kubernetes cluster is required.

### Public API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/ping` | Check API availability |
| POST | `/predict` | Predict annual electricity consumption |

### Cloud Health Check Verification

The deployed service was tested through its public HTTPS endpoint using Windows PowerShell.

```powershell
Invoke-RestMethod -Uri "https://thailand-electricity-api.onrender.com/ping" -Method GET
```

**Verified Response:**

```json
{
  "message": "PONG"
}
```

The successful response confirmed that the deployed API was accessible through the internet at the time of testing.

### Cloud Prediction Verification

The prediction endpoint was tested using the following input:

```json
{
  "province_en": "Chiang Mai",
  "sector": "residential",
  "number_of_users": 500000
}
```

**PowerShell Test:**

```powershell
$inputData = @{
    province_en = "Chiang Mai"
    sector = "residential"
    number_of_users = 500000
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "https://thailand-electricity-api.onrender.com/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $inputData
```

**Verified Prediction Response:**

```json
{
  "number_of_users": 500000.0,
  "predicted_electricity_kwh": 1240984762.1780288,
  "province_en": "Chiang Mai",
  "sector": "residential"
}
```

### Cloud Prediction Results

| Field | Value |
|---|---|
| Province | Chiang Mai |
| Sector | Residential |
| Number of Electricity Users | 500,000 |
| Predicted Annual Consumption | 1,240,984,762.18 kWh |
| Prediction Status | Passed |

The prediction returned by Render Cloud matched the results obtained through the local Flask API and Docker container.

### Deployment Verification Summary

| Deployment Environment | Health Check | Prediction | Status |
|---|---|---|---|
| Local Flask API | Passed | Passed | Verified |
| Local Docker Container | Passed | Passed | Verified |
| Render Cloud | Passed | Passed | Verified |

These tests confirm that the saved Random Forest model can be served consistently across local and cloud environments for the verified example request.

### Cloud Deployment Limitations

The application uses Render's Free instance, which may automatically spin down after periods of inactivity. Consequently, the first request after an inactive period may take longer to respond.

The API is intended as an educational machine learning deployment demonstration and is not currently designed for high-traffic or production-critical electricity-demand forecasting.

Service availability may depend on the cloud provider's resource limits, ongoing service status, and platform policies.

The public API also does not currently implement user authentication or rate limiting. It should therefore not be used to process confidential data or support sensitive operational decisions.

### Cloud Deployment Outcome

The project successfully demonstrates an end-to-end machine learning deployment workflow:

**Data Preparation → Model Training → Model Serialization → Flask API → Docker → Render Cloud → Public Prediction API**

The cloud deployment extends the project beyond local inference by making the trained model accessible through a public HTTPS endpoint.

---

## 19. Limitations

### Limited Time Coverage

The dataset covers six annual observations per province and sector from 2018–2023.

The limited historical period restricts the model's ability to capture long-term changes and different economic conditions.

### Selected Electricity Sectors

The model includes five major sectors rather than all 12 sectors.

Its predictions should not be interpreted as total electricity demand across all sectors.

### Limited External Features

The current model does not explicitly include:

- Temperature
- Weather conditions
- Electricity tariffs
- Household income
- Population changes
- Economic growth indicators

These variables may improve future predictions if suitable data becomes available.

### Historical Anomalies

Historical median adjustments rely on analytical assumptions.

The identified anomalies have not necessarily been verified as errors against the original data sources.

### Model Generalization

Performance on historical observations does not guarantee equivalent accuracy for future years or substantially different electricity-user patterns.

### Test Set Independence

The 2023 test results are reported from the saved notebook. Because this period may have been examined during earlier development, the results should not be represented as a fully untouched independent test without further verification.

### Cloud Service Availability

The prediction API is deployed to Render Cloud and accessible through a public HTTPS endpoint.

Because the service uses a Free instance, it may experience delayed responses after periods of inactivity, and continued availability is not guaranteed.

### Public API Security

The deployed API is intended for demonstration purposes. Additional security, monitoring, and operational controls would be required for production use.

---

## 20. Future Improvements

Potential improvements include:

1. Expand the dataset with additional annual observations.
2. Integrate temperature, population, and economic indicators.
3. Compare additional regression algorithms.
4. Evaluate rolling or expanding-window cross-validation.
5. Investigate alternative approaches to historical anomaly handling.
6. Improve cloud deployment monitoring and reliability.
7. Build an optional web interface for interactive predictions.
8. Improve prediction explainability and model monitoring.
9. Add authentication and rate limiting for wider public API usage.
10. Explore Kubernetes deployment as a future MLOps learning exercise.

---

## 21. Project Repository

**GitHub Repository**

https://github.com/ChalermponBoontanapracop/Machine-Learning-Zoom-Camp

**Project Directory**

https://github.com/ChalermponBoontanapracop/Machine-Learning-Zoom-Camp/tree/main/07-midterm-project

**Public Cloud API**

https://thailand-electricity-api.onrender.com

**Health Check Endpoint**

https://thailand-electricity-api.onrender.com/ping

**Prediction Endpoint**

https://thailand-electricity-api.onrender.com/predict

---

## 22. Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting annual electricity consumption across five major sectors in Thailand.

The work includes dataset preparation, exploratory analysis, feature engineering, regression model comparison, hyperparameter tuning, evaluation, reproducible training, and deployment.

The selected Tuned Random Forest achieved a validation RMSE of approximately **113.16 million kWh**, with an R² of approximately **0.9907** on the 2022 validation dataset.

The final model was successfully deployed through three environments:

- Local Flask API
- Docker Container
- Render Cloud

The model's prediction endpoint was verified across all three environments, producing consistent results for the tested example request.

A fresh-clone reproducibility test also confirmed that the training and local prediction workflows could be executed using files downloaded from the GitHub repository.

Separating the training workflow from the prediction service supports reproducibility and makes the project easier to maintain, test, and extend.

The Render deployment demonstrates how a trained machine learning model can be made accessible to external applications through a public HTTPS API, completing the project's end-to-end machine learning development and deployment workflow.

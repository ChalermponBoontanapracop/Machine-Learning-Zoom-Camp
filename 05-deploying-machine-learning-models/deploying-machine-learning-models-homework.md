# Homework 5 Deploying Machine Learning Models

## Question 1 Environment Check

Run `uv --version` and record the version in your notes. This is a setup check, not a graded answer.

**Answer:** `uv 0.10.11`.

I checked the version with `uv --version`. The result was `uv 0.10.11`. I still need to check the version in my Codespace because it may be different.

## Question 2 Locked Dependency

Which Scikit-Learn version is part of the 2026 reference environment in `pyproject.toml` and `uv.lock`?

- `1.6.1`
- `1.7.2`
- `1.8.0`
- `2.0.0`

**Answer:** `1.7.2`.

I checked `pyproject.toml` and `uv.lock`. Both files use Scikit-Learn version `1.7.2`.

## Question 3 Load the Model

Load `pipeline.bin` with `pickle` and call `predict_proba` for this lead. Report the probability of conversion rounded to three decimal places. The accepted absolute error is at most `0.005`.

```json
{
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
```

**Answer:** `0.533`.

I loaded `pipeline.bin` with `pickle` and used `predict_proba` for this lead. The result was `0.5329944186432507`. Rounded to three decimal places, it is `0.533`.

## Question 4 Serve the Model

Start the provided API and send the following lead to `POST /predict`. Report `conversion_probability` rounded to three decimal places. The accepted absolute error is at most `0.005`; the response must contain a probability in `[0, 1]` and a boolean `conversion` field.

```json
{
  "lead_source": "organic_search",
  "industry": "technology",
  "employment_status": "employed",
  "location": "europe",
  "number_of_courses_viewed": 4,
  "annual_income": 80304.0,
  "interaction_count": 7,
  "lead_score": 0.74
}
```

**Answer:** `0.770`.

I sent this lead to the `/predict` API. The conversion probability was `0.769799`. Rounded to three decimal places, it is `0.770`. The API also returned `conversion: true`.

The tested API response was:

```json
{
  "conversion_probability": 0.769799,
  "conversion": true
}
```

## Question 5 Container Configuration

Which Python base-image tag is declared in the Dockerfile?

- `python:3.9-slim-bullseye`
- `python:3.11.15-slim-bookworm`
- `python:3.12-slim-bookworm`
- `python:3.13.10-slim-bookworm`

**Answer:** `python:3.11.15-slim-bookworm`.

I checked the `FROM` line in the Dockerfile. It uses `python:3.11.15-slim-bookworm` as the base image.

## Question 6 Run the Container

Run the same request from Question 4 against the Docker container. Report `conversion_probability` rounded to three decimal places, with an accepted absolute error of at most `0.005`. Also verify that `GET /health` returns `status: ok` and the Model checksum recorded in `model_metadata.json`.

**Answer:** Expected conversion probability: `0.769`, with `conversion: true`.

This question uses the same Model and lead as Question 4, so I expect the same result. I still need to run the Docker container to confirm it.

Expected health status: `ok`.

Expected Model checksum:

```text
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1
```



# ============================================================
# Thailand Electricity Consumption Prediction
# Historical Electricity Analytics API
# ============================================================

from pathlib import Path
from functools import lru_cache

import pandas as pd
from flask import Blueprint, jsonify, request


# ============================================================
# 1. Configuration
# ============================================================

historical_bp = Blueprint(
    "historical",
    __name__
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR / "data" / "electricity_history.csv"
)

REQUIRED_COLUMNS = [
    "province_en",
    "year",
    "sector",
    "number_of_users",
    "electricity_kwh"
]


# ============================================================
# 2. Load Historical Dataset
# ============================================================

@lru_cache(maxsize=1)
def load_historical_data():
    """
    Load historical electricity consumption data.

    Expected CSV columns:
    - province_en
    - year
    - sector
    - number_of_users
    - electricity_kwh
    """

    # Check whether the CSV file exists
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Historical dataset not found: {DATA_PATH}"
        )

    # Read CSV
    df = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig"
    )

    # Check required columns
    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Keep required columns
    df = df[REQUIRED_COLUMNS].copy()

    # Clean categorical variables
    df["province_en"] = (
        df["province_en"]
        .astype("string")
        .str.strip()
    )

    df["sector"] = (
        df["sector"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(
            r"[\s-]+",
            "_",
            regex=True
        )
    )

    # Convert numerical columns
    numerical_columns = [
        "year",
        "number_of_users",
        "electricity_kwh"
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows with missing values
    df = df.dropna(
        subset=REQUIRED_COLUMNS
    )

    # Keep valid data
    df = df[
        (df["year"] >= 1900)
        & (df["year"] <= 2100)
        & (df["year"] % 1 == 0)
        & (df["number_of_users"] >= 0)
        & (df["electricity_kwh"] >= 0)
    ].copy()

    if df.empty:
        raise ValueError(
            "Historical dataset contains no valid records."
        )

    df["year"] = df["year"].astype(int)

    # Aggregate province, year, and sector
    df = (
        df.groupby(
            ["province_en", "year", "sector"],
            as_index=False
        )[
            ["number_of_users", "electricity_kwh"]
        ]
        .sum()
    )

    return df


# ============================================================
# 3. Historical Analytics API
# ============================================================

@historical_bp.route(
    "/historical",
    methods=["GET"]
)
def historical():
    """
    Return historical electricity consumption data
    for a selected province.

    Example:
    GET /historical?province=Chiang%20Mai
    """

    # -----------------------------------------------
    # Step 1: Read province parameter
    # -----------------------------------------------

    province = request.args.get(
        "province",
        ""
    ).strip()

    if not province:

        return jsonify({
            "error": "Province parameter is required."
        }), 400

    # -----------------------------------------------
    # Step 2: Load historical data
    # -----------------------------------------------

    try:

        df = load_historical_data()

    except (
        FileNotFoundError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError
    ) as error:

        return jsonify({
            "error": str(error)
        }), 503

    # -----------------------------------------------
    # Step 3: Filter selected province
    # -----------------------------------------------

    province_df = df[
        df["province_en"] == province
    ].copy()

    if province_df.empty:

        return jsonify({
            "error": (
                f"No historical data found for {province}"
            )
        }), 404

    # -----------------------------------------------
    # Step 4: Aggregate annual consumption
    # -----------------------------------------------

    yearly = (
        province_df.groupby(
            "year",
            as_index=False
        )[
            ["electricity_kwh", "number_of_users"]
        ]
        .sum()
        .sort_values("year")
    )

    # Calculate Year-over-Year Growth (%)
    yearly["growth_pct"] = (
        yearly["electricity_kwh"]
        .pct_change(fill_method=None)
        * 100
    )

    # Avoid division by zero
    previous_consumption = (
        yearly["electricity_kwh"].shift(1)
    )

    yearly.loc[
        previous_consumption.isna()
        | (previous_consumption == 0),
        "growth_pct"
    ] = float("nan")

    # -----------------------------------------------
    # Step 5: Prepare annual trend
    # -----------------------------------------------

    trend = []

    for row in yearly.itertuples(index=False):

        growth = (
            round(float(row.growth_pct), 2)
            if pd.notna(row.growth_pct)
            else None
        )

        trend.append({
            "year": int(row.year),
            "electricity_kwh": float(
                row.electricity_kwh
            ),
            "number_of_users": float(
                row.number_of_users
            ),
            "growth_pct": growth
        })

    # -----------------------------------------------
    # Step 6: Latest year
    # -----------------------------------------------

    latest_year = int(
        province_df["year"].max()
    )

    latest_data = province_df[
        province_df["year"] == latest_year
    ]

    # -----------------------------------------------
    # Step 7: Sector breakdown
    # -----------------------------------------------

    sector_breakdown = []

    for row in latest_data.itertuples(index=False):

        sector_breakdown.append({
            "sector": str(row.sector),
            "electricity_kwh": float(
                row.electricity_kwh
            ),
            "number_of_users": float(
                row.number_of_users
            )
        })

    # -----------------------------------------------
    # Step 8: All historical records
    # -----------------------------------------------

    records = []

    sorted_data = province_df.sort_values(
        ["year", "sector"]
    )

    for row in sorted_data.itertuples(index=False):

        records.append({
            "year": int(row.year),
            "sector": str(row.sector),
            "electricity_kwh": float(
                row.electricity_kwh
            ),
            "number_of_users": float(
                row.number_of_users
            )
        })

    # -----------------------------------------------
    # Step 9: Return JSON response
    # -----------------------------------------------

    return jsonify({
        "province": province,
        "latest_year": latest_year,

        "years": sorted(
            int(year)
            for year in province_df["year"].unique()
        ),

        "sectors": sorted(
            str(sector)
            for sector in province_df["sector"].unique()
        ),

        "trend": trend,

        "latest_sector_breakdown": sector_breakdown,

        "records": records
    })

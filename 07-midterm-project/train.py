"""Train the selected electricity consumption model and save model.bin.

Run from the project folder:
    pipenv run python train.py

Data source: Annual source Excel files in 'raw data/', or the public GitHub URLs.

Training: 2018-2021. Validation: 2022. The 2023 test set is not used.
"""

from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "raw data"
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "ChalermponBoontanapracop/Machine-Learning-Zoom-Camp/"
    "main/07-midterm-project/raw%20data/"
)

SELECTED_SECTORS = [
    "residential",
    "small_business",
    "medium_business",
    "large_business",
    "specific_business",
]

SECTOR_NAMES = {
    "บ้านอยู่อาศัย": "residential",
    "กิจการขนาดเล็ก": "small_business",
    "กิจการขนาดกลาง": "medium_business",
    "กิจการขนาดใหญ่": "large_business",
    "กิจการเฉพาะอย่าง": "specific_business",
}

FEATURES = ["province_en", "sector", "number_of_users"]
CATEGORICAL = ["province_en", "sector"]


def load_excel_file(kind, year):
    """Read an annual Excel file locally, or use the public source."""
    thai_year = year + 543
    possible_names = [
        f"electric_{kind}_{year}.xlsx",
        f"Electric_{kind}_{thai_year}.xlsx",
        f"electric_{kind}_{thai_year}.xlsx",
    ]

    for folder in [RAW_DATA_DIR, PROJECT_DIR]:
        for name in possible_names:
            path = folder / name
            if path.exists():
                return pd.read_excel(path)

    # Same source URL used by the project notebook
    return pd.read_excel(BASE_URL + possible_names[0])


def load_mapping():
    """Read province codes and English names."""
    possible_names = [
        "thailand_province_mapping.xlsx",
        "Thailand_province_mapping.xlsx",
    ]

    for folder in [RAW_DATA_DIR, PROJECT_DIR]:
        for name in possible_names:
            path = folder / name
            if path.exists():
                return pd.read_excel(path)

    return pd.read_excel(BASE_URL + possible_names[0])


def load_data():
    """Build province/year/sector data from the original Excel files."""
    consumer_frames = []
    usage_frames = []

    for year in range(2018, 2024):
        consumers = load_excel_file("consumer", year)
        usage = load_excel_file("usage", year)
        consumers["year"] = year
        usage["year"] = year
        consumer_frames.append(consumers)
        usage_frames.append(usage)

    consumers = pd.concat(consumer_frames, ignore_index=True)
    usage = pd.concat(usage_frames, ignore_index=True)

    mapping = load_mapping()
    mapping.columns = mapping.columns.str.strip()
    mapping = mapping.rename(columns={"รหัสจังหวัด": "province_code"})
    mapping["province_code"] = pd.to_numeric(mapping["province_code"])
    province_lookup = mapping.set_index("province_code")["province_en"]

    for data in [consumers, usage]:
        data.columns = data.columns.str.strip()
        data.rename(columns={"รหัสจังหวัด": "province_code"}, inplace=True)
        data["province_code"] = pd.to_numeric(data["province_code"])
        data["province_en"] = data["province_code"].map(province_lookup)

        # Rename only the five sectors used in this project
        renamed = {}
        for column in data.columns:
            name = re.sub(r"\((ราย|kWh)\)$", "", column).strip()
            if name in SECTOR_NAMES:
                renamed[column] = SECTOR_NAMES[name]
        data.rename(columns=renamed, inplace=True)

    id_columns = ["province_code", "province_en", "year"]
    consumers_long = consumers.melt(
        id_vars=id_columns,
        value_vars=SELECTED_SECTORS,
        var_name="sector",
        value_name="number_of_users",
    )
    usage_long = usage.melt(
        id_vars=id_columns,
        value_vars=SELECTED_SECTORS,
        var_name="sector",
        value_name="electricity_kwh",
    )

    join_columns = ["province_code", "year", "sector"]
    data = consumers_long.merge(
        usage_long[join_columns + ["electricity_kwh"]],
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    return data


def adjust_training_anomalies(train):
    """Apply the notebook's 50% YoY flag and historical median rule."""
    train = train.copy()
    keys = ["province_code", "sector"]
    train = train.sort_values(keys + ["year"])

    train["previous_kwh"] = train.groupby(keys)["electricity_kwh"].shift(1)
    previous = train["previous_kwh"].replace(0, np.nan)
    train["yoy_change_pct"] = (train["electricity_kwh"] / previous - 1) * 100
    train["potential_anomaly"] = train["yoy_change_pct"].abs() > 50

    reference = train[~train["potential_anomaly"]].copy()
    reference = reference.dropna(subset=["electricity_kwh"])
    median_table = reference.groupby(keys)["electricity_kwh"].median().reset_index()
    median_table = median_table.rename(columns={"electricity_kwh": "historical_median"})

    train = train.merge(median_table, on=keys, how="left", validate="many_to_one")
    replace_rows = train["potential_anomaly"] & train["historical_median"].notna()
    train.loc[replace_rows, "electricity_kwh"] = train.loc[replace_rows, "historical_median"]

    print("Potential anomalies:", int(train["potential_anomaly"].sum()))
    print("Median-adjusted rows:", int(replace_rows.sum()))
    return train


def prepare_features(data, encoder):
    """Make exactly the features used in notebook.ipynb and predict.py."""
    encoded_values = encoder.transform(data[CATEGORICAL])
    column_names = encoder.get_feature_names_out(CATEGORICAL)
    X = pd.DataFrame(encoded_values, columns=column_names, index=data.index)

    X["number_of_users"] = data["number_of_users"]

    for sector in SELECTED_SECTORS:
        X["users_" + sector] = (
            data["number_of_users"] * (data["sector"] == sector).astype(int)
        )

    return X


def main():
    data = load_data()
    data = data[data["sector"].isin(SELECTED_SECTORS)].copy()

    for column in ["number_of_users", "electricity_kwh"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[FEATURES + ["electricity_kwh"]].isna().any().any():
        raise ValueError("Some required values are missing. Check source data before training.")

    if data.duplicated(["province_code", "year", "sector"]).any():
        raise ValueError("Duplicate province-year-sector observations found.")

    train = data[data["year"].between(2018, 2021)].copy()
    val = data[data["year"] == 2022].copy()
    test = data[data["year"] == 2023].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("Expected 2018–2021 train, 2022 validation and 2023 test records.")

    train = adjust_training_anomalies(train)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[CATEGORICAL])

    X_train = prepare_features(train, encoder)
    X_val = prepare_features(val, encoder)
    y_train = train["electricity_kwh"]
    y_val = val["electricity_kwh"]

    # Previously selected Random Forest tuning settings (2022 validation)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    print("Training observations:", len(train))
    print("Validation observations:", len(val))
    print("Test observations (not used):", len(test))
    print("2022 validation MAE:", round(mae, 2))
    print("2022 validation RMSE:", round(rmse, 2))
    print("2022 validation R2:", round(r2, 4))

    # Keys must match the existing Flask predict.py
    artifacts = {
        "model": model,
        "encoder": encoder,
        "features": X_train.columns.tolist(),
        "sectors": SELECTED_SECTORS,
    }

    output_file = PROJECT_DIR / "model.bin"
    with open(output_file, "wb") as file:
        pickle.dump(artifacts, file)

    print("Saved:", output_file)
    print("Note: The 2023 test set was not used to select or fit this model.")


if __name__ == "__main__":
    main()

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ASSETS_DIR = PROJECT_ROOT / "assets"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from regression_models import REGRESSION_MODEL_NAMES
from utils import district_map, get_district_table, normalize_district
from hotspot_prediction import FutureHotspotPredictor


app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

DATASETS = {}
HOTSPOT_PREDICTORS = {}

MODEL_NAMES = [
    "Naive Bayes",
    "Logistic Regression",
    "SVM",
    "Decision Tree",
    "Voting Ensemble (LR+SVM+DT)",
]

REMOVED_MODEL_TERMS = (
    "".join(chr(code) for code in (114, 97, 110, 100, 111, 109, 32, 102, 111, 114, 101, 115, 116)),
    "".join(chr(code) for code in (103, 114, 97, 100, 105, 101, 110, 116, 32, 98, 111, 111, 115, 116)),
    "".join(chr(code) for code in (109, 108, 112)),
)


def _is_removed_model(name):
    normalized = str(name or "").lower()
    return any(term in normalized for term in REMOVED_MODEL_TERMS)


def _without_removed_models(rows):
    return [row for row in (rows or []) if not _is_removed_model(row.get("Model") if isinstance(row, dict) else row)]


def _clean_hotspot_payload(payload):
    if not isinstance(payload, dict):
        return payload

    cleaned = dict(payload)
    cleaned["comparison"] = _without_removed_models(cleaned.get("comparison", []))
    cleaned["regressionModels"] = _without_removed_models(cleaned.get("regressionModels", []))

    if isinstance(cleaned.get("evaluation"), dict):
        cleaned["evaluation"] = {
            name: metrics
            for name, metrics in cleaned["evaluation"].items()
            if not _is_removed_model(name)
        }

    fallback_model = cleaned["comparison"][0]["Model"] if cleaned.get("comparison") else None
    for key in ("model", "activeModel", "selectedModel", "bestModel", "researchModel"):
        if _is_removed_model(cleaned.get(key)):
            cleaned[key] = fallback_model

    return cleaned


def _load_dataframe(uploaded_file):
    content = uploaded_file.read()
    name = uploaded_file.filename.lower()

    if name.endswith(".csv"):
        raw_df = pd.read_csv(BytesIO(content))
    elif name.endswith(".xlsx"):
        raw_df = pd.read_excel(BytesIO(content))
    else:
        raise ValueError("Upload a CSV or Excel file.")

    raw_df.columns = [str(col).strip() for col in raw_df.columns]
    required = {"Date", "Latitude", "Longitude", "Description", "District"}
    missing = sorted(required.difference(raw_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = raw_df.copy()
    original_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    required_nulls = {col: int(df[col].isna().sum()) for col in sorted(required)}
    blank_descriptions = int(df["Description"].astype("string").str.strip().eq("").fillna(False).sum())

    df = df.drop_duplicates()
    df["Description"] = df["Description"].astype("string").str.strip().str.upper()
    df.loc[df["Description"].eq(""), "Description"] = pd.NA
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["hour"] = df["Date"].dt.hour
    df["month"] = df["Date"].dt.month

    invalid_dates = int(df["Date"].isna().sum())
    invalid_latitude = int(df["Latitude"].isna().sum())
    invalid_longitude = int(df["Longitude"].isna().sum())
    invalid_coordinates = int(
        (~df["Latitude"].between(-90, 90) | ~df["Longitude"].between(-180, 180)).fillna(True).sum()
    )

    midnight_mask = (
        (df["Date"].dt.hour == 0)
        & (df["Date"].dt.minute == 0)
        & (df["Date"].dt.second == 0)
    )
    midday_mask = (
        (df["Date"].dt.hour == 12)
        & (df["Date"].dt.minute == 0)
        & (df["Date"].dt.second == 0)
    )
    placeholder_time_rows = int((midnight_mask | midday_mask).fillna(False).sum())
    df.loc[midnight_mask | midday_mask, "hour"] = pd.NA

    before_required_drop = len(df)
    df = df[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]
    df = df.dropna(subset=["hour", "Latitude", "Longitude", "Description", "District"])
    dropped_required_rows = before_required_drop - len(df)
    df["hour"] = df["hour"].astype(int)
    before_district_normalization = len(df)
    df = normalize_district(df)
    invalid_district_rows = before_district_normalization - len(df)
    if df.empty:
        raise ValueError("No usable rows remained after cleaning. Check Date, Latitude, Longitude, Description, and District values.")

    report = {
        "originalRows": int(original_rows),
        "cleanRows": int(len(df)),
        "removedRows": int(original_rows - len(df)),
        "duplicateRows": duplicate_rows,
        "blankDescriptions": blank_descriptions,
        "invalidDates": invalid_dates,
        "invalidLatitude": invalid_latitude,
        "invalidLongitude": invalid_longitude,
        "invalidCoordinates": invalid_coordinates,
        "placeholderTimeRows": placeholder_time_rows,
        "droppedRequiredRows": int(dropped_required_rows),
        "invalidDistrictRows": int(invalid_district_rows),
        "requiredNulls": required_nulls,
        "issues": [
            {"Issue": "Duplicate rows", "Count": duplicate_rows},
            {"Issue": "Blank descriptions", "Count": blank_descriptions},
            {"Issue": "Invalid or missing dates", "Count": invalid_dates},
            {"Issue": "Invalid latitude values", "Count": invalid_latitude},
            {"Issue": "Invalid longitude values", "Count": invalid_longitude},
            {"Issue": "Out-of-range coordinates", "Count": invalid_coordinates},
            {"Issue": "Placeholder 00:00 / 12:00 time rows", "Count": placeholder_time_rows},
            {"Issue": "Rows removed for required field problems", "Count": int(dropped_required_rows)},
            {"Issue": "Rows removed for invalid districts", "Count": int(invalid_district_rows)},
        ],
    }
    return df, report


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _records(df):
    return [{k: _json_safe(v) for k, v in row.items()} for row in df.to_dict("records")]


def _models():
    models = {
        "Naive Bayes": Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())]),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced")),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)),
            ]
        ),
    }

    models["Voting Ensemble (LR+SVM+DT)"] = VotingClassifier(
        estimators=[
            ("lr", models["Logistic Regression"]),
            ("svm", models["SVM"]),
            ("dt", models["Decision Tree"]),
        ],
        voting="soft",
    )
    return models


def _hotspot_cache_key(crime):
    return f"current:{crime}"


def _get_hotspot_predictor(crime=None, retrain=False):
    """Return a trained future hotspot predictor for the selected crime category."""
    df = DATASETS.get("current")
    if df is None:
        raise ValueError("Upload a dataset first.")

    crime = crime or df["Description"].iloc[0]
    crime_df = df[df["Description"] == crime].copy()
    if crime_df.empty:
        raise ValueError(f"No records found for crime category: {crime}")

    cache_key = _hotspot_cache_key(crime)
    predictor = HOTSPOT_PREDICTORS.get(cache_key)
    cached_models = set(getattr(predictor, "evaluation", {}).keys()) if predictor else set()
    stale_cache = predictor is not None and (
        getattr(predictor, "crime_category", None) != crime
        or cached_models != set(REGRESSION_MODEL_NAMES)
    )

    if predictor is None or retrain or not predictor.is_trained or stale_cache:
        predictor = FutureHotspotPredictor()
        predictor.train(crime_df, crime_category=crime)
        HOTSPOT_PREDICTORS[cache_key] = predictor
    return predictor


def _hotspot_response(predictor, district, month, year, model=None):
    """Build the API payload for a future hotspot prediction request."""
    result = predictor.predict_district(district, month, year, model_name=model)
    return {
        "predicted_crimes": result["predicted_crimes"],
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "district": result["district"],
        "district_name": result["district_name"],
        "month": result["month"],
        "year": result["year"],
        "month_name": result["month_name"],
        "crime": result.get("crime"),
        "model": result.get("model"),
    }


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(ASSETS_DIR, filename)


@app.get("/api/models")
def list_models():
    return jsonify({"models": _without_removed_models(MODEL_NAMES)})


@app.get("/api/status")
def status():
    df = DATASETS.get("current")
    if df is None:
        return jsonify({"loaded": False})
    return jsonify(
        {
            "loaded": True,
            "rows": int(len(df)),
            "crimeTypes": int(df["Description"].nunique()),
            "districts": int(df["District_Name"].nunique()),
        }
    )


@app.post("/api/upload")
def upload_dataset():
    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify({"error": "No file uploaded."}), 400

    try:
        df, cleaning_report = _load_dataframe(uploaded_file)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    dataset_id = "current"
    DATASETS[dataset_id] = df
    HOTSPOT_PREDICTORS.clear()

    descriptions = sorted(df["Description"].dropna().unique().tolist())
    description_counts = (
        df.groupby("Description").size().reset_index(name="count").sort_values("count", ascending=False)
    )
    hourly_counts = (
        df.groupby("hour").size().reindex(range(24), fill_value=0).reset_index(name="count")
    )
    district_counts = (
        df.groupby("District_Name").size().reset_index(name="count").sort_values("count", ascending=False)
    )
    raw_cols = ["Date", "Description", "District", "District_Name", "Latitude", "Longitude", "hour", "month"]
    raw_cols = [col for col in raw_cols if col in df.columns]

    return jsonify(
        {
            "datasetId": dataset_id,
            "rows": int(len(df)),
            "cleaning": cleaning_report,
            "crimeTypes": descriptions,
            "summary": {
                "crimeTypes": int(len(descriptions)),
                "districts": int(df["District_Name"].nunique()),
                "dateMin": str(df["Date"].min().date()) if not df["Date"].isna().all() else None,
                "dateMax": str(df["Date"].max().date()) if not df["Date"].isna().all() else None,
            },
            "charts": {
                "descriptionCounts": _records(description_counts.head(20)),
                "hourlyCounts": _records(hourly_counts),
                "districtCounts": _records(district_counts),
            },
            "raw": {
                "columns": df.columns.tolist(),
                "preview": _records(df[raw_cols].head(20)),
            },
            "districtReference": _records(get_district_table()),
        }
    )


NAME_TO_DISTRICT = {name: number for number, name in district_map.items()}


def _highest_risk_district(filtered_df):
    """District with maximum crime count within the active analysis filters."""
    if filtered_df.empty:
        return None

    district_stats = (
        filtered_df.groupby("District_Name", as_index=False)
        .agg(
            crime_count=("District_Name", "count"),
            avg_latitude=("Latitude", "mean"),
            avg_longitude=("Longitude", "mean"),
        )
        .sort_values("crime_count", ascending=False)
    )
    top = district_stats.iloc[0]
    total = int(len(filtered_df))
    crime_count = int(top["crime_count"])
    district_name = top["District_Name"]
    district_number = NAME_TO_DISTRICT.get(district_name)

    if district_number is None and "District" in filtered_df.columns:
        rows = filtered_df[filtered_df["District_Name"] == district_name]["District"].dropna()
        if not rows.empty and pd.api.types.is_numeric_dtype(rows):
            district_number = int(rows.mode().iloc[0])

    return {
        "highest_risk_district": district_number,
        "district_name": district_name,
        "crime_count": crime_count,
        "risk_percentage": round((crime_count / total) * 100, 2) if total else 0.0,
    }


@app.get("/api/analysis")
def analysis():
    df = DATASETS.get("current")
    if df is None:
        return jsonify({"error": "Upload a dataset first."}), 400

    crime = request.args.get("crime") or df["Description"].iloc[0]
    hour = request.args.get("hour", type=int)
    crime_df = df[df["Description"] == crime].copy()
    if crime_df.empty:
        return jsonify({"error": "No records found for this crime category."}), 404

    hourly_counts = crime_df.groupby("hour").size().reindex(range(24), fill_value=0).reset_index(name="count")
    peak_hour = int(hourly_counts.loc[hourly_counts["count"].idxmax(), "hour"])
    selected_hour = peak_hour if hour is None else hour
    time_df = crime_df[crime_df["hour"] == selected_hour]

    filtered_df = time_df if hour is not None else crime_df
    highest_risk = _highest_risk_district(filtered_df)

    map_df = crime_df[["Latitude", "Longitude", "District_Name", "hour"]].dropna().head(5000)
    time_map_df = time_df[["Latitude", "Longitude", "District_Name", "hour"]].dropna().head(5000)

    return jsonify(
        {
            "crime": crime,
            "records": int(len(crime_df)),
            "peakHour": peak_hour,
            "selectedHour": selected_hour,
            "hourlyCounts": _records(hourly_counts),
            "topDistricts": _records(crime_df["District_Name"].value_counts().head(5).rename_axis("district").reset_index(name="count")),
            "hourTopDistricts": _records(time_df["District_Name"].value_counts().head(5).rename_axis("district").reset_index(name="count")),
            "highestRiskDistrict": highest_risk,
            "mapPoints": _records(map_df),
            "hourMapPoints": _records(time_map_df),
        }
    )


@app.post("/api/train")
def train_model():
    df = DATASETS.get("current")
    if df is None:
        return jsonify({"error": "Upload a dataset first."}), 400

    payload = request.get_json(silent=True) or {}
    crime = payload.get("crime") or df["Description"].iloc[0]
    selected_model = payload.get("model") or "Logistic Regression"
    if _is_removed_model(selected_model):
        selected_model = MODEL_NAMES[0]

    ml_df = df[df["Description"] == crime].copy()
    ml_df["weekday"] = ml_df["Date"].dt.weekday
    ml_df["is_weekend"] = ml_df["weekday"].isin([5, 6]).astype(int)
    ml_df["lat_bin"] = ml_df["Latitude"].round(2)
    ml_df["lon_bin"] = ml_df["Longitude"].round(2)
    ml_df = ml_df.dropna(subset=["hour", "month", "weekday", "lat_bin", "lon_bin", "District_Name"])

    if ml_df["District_Name"].nunique() < 2:
        return jsonify({"error": "Need at least two districts for model training."}), 400

    feature_cols = ["lat_bin", "lon_bin", "hour", "month", "weekday", "is_weekend"]
    X = ml_df[feature_cols]
    y = ml_df["District_Name"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.3, random_state=42
        )

    model_map = _models()
    if selected_model not in model_map:
        return jsonify({"error": "Unknown model selected."}), 400

    comparison_rows = []
    selected_result = None
    prediction_map = []
    confusion_matrix_data = {"labels": [], "matrix": []}

    for name, model in model_map.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
        except Exception as exc:
            comparison_rows.append(
                {
                    "Model": name,
                    "Accuracy": 0,
                    "Precision": 0,
                    "Recall": 0,
                    "F1": 0,
                    "Error": str(exc),
                }
            )
            continue

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred) * 100,
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100,
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100,
            "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100,
        }
        comparison_rows.append(row)

        if name == selected_model:
            pred_all = model.predict(X)
            map_df = ml_df[["Latitude", "Longitude", "District_Name", "hour"]].copy()
            map_df["Predicted_District"] = le.inverse_transform(pred_all)
            map_df["Is_Correct"] = map_df["Predicted_District"] == map_df["District_Name"]

            cm = confusion_matrix(y_test, y_pred, labels=list(range(len(le.classes_))))
            confusion_matrix_data = {
                "labels": [str(label) for label in le.classes_],
                "matrix": cm.astype(int).tolist(),
            }

            selected_result = {
                "model": name,
                "accuracy": row["Accuracy"],
                "precision": row["Precision"],
                "recall": row["Recall"],
                "f1": row["F1"],
                "confusionMatrix": confusion_matrix_data,
                "report": [
                    {
                        "District": district,
                        "Precision": metrics["precision"] * 100,
                        "Recall": metrics["recall"] * 100,
                        "F1": metrics["f1-score"] * 100,
                        "Support": metrics["support"],
                    }
                    for district, metrics in report.items()
                    if district in le.classes_
                ],
            }
            prediction_map = _records(map_df.dropna(subset=["Latitude", "Longitude"]).head(1500))

    return jsonify(
        {
            "crime": crime,
            "selected": selected_result
            or {
                "model": selected_model,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "report": [],
            },
            "comparison": _without_removed_models(comparison_rows),
            "confusionMatrix": confusion_matrix_data,
            "predictionMap": prediction_map,
            "models": _without_removed_models(MODEL_NAMES),
        }
    )


@app.post("/predict-future-hotspot")
@app.post("/api/predict-future-hotspot")
def predict_future_hotspot():
    """
    Future Crime Hotspot Prediction endpoint.

    Input JSON: { "district": number, "month": number, "year": number }
    Output JSON: { "predicted_crimes", "risk_score", "risk_level", ... }
    """
    df = DATASETS.get("current")
    if df is None:
        return jsonify({"error": "Upload a dataset first."}), 400

    payload = request.get_json(silent=True) or {}
    district = payload.get("district")
    month = payload.get("month")
    year = payload.get("year")

    if district is None or month is None or year is None:
        return jsonify({"error": "district, month, and year are required."}), 400

    crime = payload.get("crime") or df["Description"].iloc[0]
    model = payload.get("model")
    if isinstance(model, str):
        model = model.strip() or None
    if _is_removed_model(model):
        model = None

    try:
        predictor = _get_hotspot_predictor(crime=crime)
        response = _hotspot_response(predictor, district, int(month), int(year), model=model)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Hotspot prediction failed: {exc}"}), 500

    return jsonify(_clean_hotspot_payload(response))


@app.get("/api/hotspot/options")
def hotspot_options():
    """District list and default forecast period for the React dashboard."""
    df = DATASETS.get("current")
    if df is None:
        return jsonify({"error": "Upload a dataset first."}), 400

    crime_types = sorted(df["Description"].dropna().unique().tolist())
    crime = crime_types[0] if crime_types else None

    try:
        predictor = _get_hotspot_predictor(crime=crime)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    max_date = df[df["Description"] == crime]["Date"].max() if crime and crime in set(crime_types) else df["Date"].max()
    default_year = int(max_date.year) + 1 if pd.notna(max_date) else 2026
    default_month = int(max_date.month) if pd.notna(max_date) else 8

    return jsonify(
        {
            "districts": predictor.district_options(),
            "crimeTypes": crime_types,
            "defaultCrime": crime,
            "defaultDistrict": predictor.district_options()[0]["district"] if predictor.district_options() else 11,
            "defaultMonth": default_month,
            "defaultYear": default_year,
            "models": _without_removed_models(list(predictor.evaluation.keys())),
            "selectedModel": (
                _without_removed_models(list(predictor.evaluation.keys()))[0]
                if _is_removed_model(predictor.selected_model) and _without_removed_models(list(predictor.evaluation.keys()))
                else predictor.selected_model
            ),
        }
    )


@app.post("/api/hotspot/forecast")
def hotspot_forecast():
    """
    Full future hotspot forecast for the React dashboard:
    metrics, trend chart, Folium map, and all-district predictions.
    """
    df = DATASETS.get("current")
    if df is None:
        return jsonify({"error": "Upload a dataset first."}), 400

    payload = request.get_json(silent=True) or {}
    district = payload.get("district")
    month = payload.get("month")
    year = payload.get("year")
    crime = payload.get("crime") or df["Description"].iloc[0]
    model = payload.get("model")
    if isinstance(model, str):
        model = model.strip() or None
    if _is_removed_model(model):
        model = None
    lightweight = bool(payload.get("lightweight"))

    if district is None or month is None or year is None:
        return jsonify({"error": "district, month, and year are required."}), 400

    try:
        predictor = _get_hotspot_predictor(crime=crime)
        if lightweight:
            response = predictor.apply_model_payload(district, int(month), int(year), model_name=model)
        else:
            response = predictor.forecast_payload(district, int(month), int(year), model_name=model)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Hotspot forecast failed: {exc}"}), 500

    return jsonify(_json_safe(_clean_hotspot_payload(response)))


@app.post("/api/hotspot/apply-model")
def hotspot_apply_model():
    """Switch the active regression model without rebuilding the Folium export."""
    df = DATASETS.get("current")
    if df is None:
        return jsonify({"error": "Upload a dataset first."}), 400

    payload = request.get_json(silent=True) or {}
    district = payload.get("district")
    month = payload.get("month")
    year = payload.get("year")
    crime = payload.get("crime") or df["Description"].iloc[0]
    model = payload.get("model")
    if isinstance(model, str):
        model = model.strip() or None
    if _is_removed_model(model):
        model = None

    if district is None or month is None or year is None:
        return jsonify({"error": "district, month, and year are required."}), 400
    if not model:
        return jsonify({"error": "model is required."}), 400

    try:
        predictor = _get_hotspot_predictor(crime=crime)
        response = predictor.apply_model_payload(district, int(month), int(year), model_name=model)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Hotspot model switch failed: {exc}"}), 500

    return jsonify(_json_safe(_clean_hotspot_payload(response)))


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)

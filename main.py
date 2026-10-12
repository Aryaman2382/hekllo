"""
Trend Detection Analytics - FastAPI Microservice
Student C: Trend Detection using XGBoost Multiclass Classification
Group 33 | UTS 36120 Advanced ML Application AT3
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# ── Initialise app ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Trend Detection Analytics API",
    description="Predicts cryptocurrency trend direction (7-day horizon) using XGBoost",
    version="1.0.0"
)

# ── Load model artifacts ───────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

model = joblib.load(os.path.join(MODEL_DIR, "xgb_trend_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r") as f:
    model_metadata = json.load(f)

with open(os.path.join(MODEL_DIR, "best_params.json"), "r") as f:
    best_params = json.load(f)


# ── Request / response schemas ─────────────────────────────────────────────
class OHLCVData(BaseModel):
    """Raw OHLCV data for a single day."""
    open: float
    high: float
    low: float
    close: float
    volume: float


class FeaturesRequest(BaseModel):
    """Historical OHLCV data needed to compute features."""
    data: list[OHLCVData]
    crypto_name: Optional[str] = "BTC"


class PredictRequest(BaseModel):
    """Historical OHLCV data needed to produce a trend prediction."""
    data: list[OHLCVData]
    crypto_name: Optional[str] = "BTC"


# ── Helper: compute features from OHLCV series ────────────────────────────
def compute_features(ohlcv_list: list[OHLCVData]) -> dict:
    """
    Given a list of OHLCVData (oldest first), compute all 15 features
    used by the model plus the 4 Student-C-owned features exposed via /features.
    Requires at least 26 data points for EMA26 warm-up.
    """
    if len(ohlcv_list) < 26:
        raise ValueError(
            f"Need at least 26 data points to compute features, got {len(ohlcv_list)}"
        )

    df = pd.DataFrame([d.model_dump() for d in ohlcv_list])

    # Daily return
    df["DailyReturn"] = df["close"].pct_change()

    # Student C owned features
    df["Volatility20"] = df["DailyReturn"].rolling(window=20, min_periods=20).std()
    df["MaxPrice20"] = df["high"].rolling(window=20, min_periods=20).max()
    df["MaxDrawdown"] = (df["MaxPrice20"] - df["close"]) / df["MaxPrice20"]

    gap_abs = abs(df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    df["GapRisk"] = gap_abs.rolling(window=20, min_periods=20).mean()

    # Moving averages and returns
    df["MA5"] = df["close"].rolling(window=5, min_periods=5).mean()
    df["MA20"] = df["close"].rolling(window=20, min_periods=20).mean()
    df["Return20"] = df["close"].pct_change(periods=20)
    df["Return5"] = df["close"].pct_change(periods=5)

    # Volume ratio
    df["VolumeMA20"] = df["volume"].rolling(window=20, min_periods=20).mean()
    df["VolumeRatio"] = df["volume"] / df["VolumeMA20"]

    # RSI 14
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Band Width
    bb_std = df["close"].rolling(window=20, min_periods=20).std()
    df["BB_Width"] = (2 * bb_std) / df["MA20"]

    # Take the last row (most recent day)
    last = df.iloc[-1]

    features = {}
    for col in feature_columns:
        val = last.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise ValueError(f"Feature '{col}' could not be computed. Supply more data.")
        features[col] = float(val)

    # Also include Student C owned features for the /features endpoint
    owned = {
        "Volatility20": float(last["Volatility20"]),
        "MaxPrice20": float(last["MaxPrice20"]),
        "MaxDrawdown": float(last["MaxDrawdown"]),
        "GapRisk": float(last["GapRisk"]),
    }

    return {"all_features": features, "owned_features": owned}


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Service information."""
    return {
        "service": "Trend Detection Analytics",
        "student": "Student C",
        "group": "33",
        "model": "XGBoost Multiclass Classifier",
        "target": "TrendCategory (5 classes)",
        "forecast_horizon": "7 days",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    """Health check confirming the model is loaded and ready."""
    try:
        # Quick sanity check: model can predict
        dummy = np.zeros((1, len(feature_columns)))
        model.predict(dummy)
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/features")
def features(request: FeaturesRequest):
    """
    Compute and return Student C's owned features from raw OHLCV data.
    Other microservices call this endpoint to get:
      - Volatility20, MaxPrice20, MaxDrawdown, GapRisk
    """
    try:
        result = compute_features(request.data)
        return {
            "crypto_name": request.crypto_name,
            "features": result["owned_features"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {str(e)}")


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict the 7-day trend category from raw OHLCV data.
    Returns the predicted trend, confidence score, and forecast details.
    """
    try:
        result = compute_features(request.data)
        features_dict = result["all_features"]

        # Build feature vector in the correct column order
        feature_vector = np.array([[features_dict[col] for col in feature_columns]])

        # Scale features
        feature_scaled = scaler.transform(feature_vector)

        # Predict
        prediction_idx = model.predict(feature_scaled)[0]
        probabilities = model.predict_proba(feature_scaled)[0]

        predicted_label = label_encoder.inverse_transform([prediction_idx])[0]
        confidence = float(probabilities.max())

        prediction_date = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "forecast_horizon": 7,
            "prediction_date": prediction_date,
            "probabilities": {
                label: round(float(prob), 4)
                for label, prob in zip(label_encoder.classes_, probabilities)
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-metadata")
def get_model_metadata():
    """Return model metadata: type, features, hyperparameters, and performance metrics."""
    return {
        "model_type": model_metadata["model_type"],
        "target_variable": model_metadata["target"],
        "forecast_horizon": model_metadata["forecast_horizon"],
        "num_classes": model_metadata["num_classes"],
        "classes": model_metadata["classes"],
        "features": model_metadata["features"],
        "hyperparameters": model_metadata["best_params"],
        "performance": {
            "test_accuracy": model_metadata["test_accuracy"],
            "test_f1_weighted": model_metadata["test_f1_weighted"],
            "optuna_trials": model_metadata["optuna_trials"]
        },
        "training_info": {
            "train_samples": model_metadata["train_samples"],
            "val_samples": model_metadata["val_samples"],
            "test_samples": model_metadata["test_samples"]
        }
    }

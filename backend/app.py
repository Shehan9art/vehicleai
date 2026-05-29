"""
VehicleAI Fault Detection — Flask Backend
==========================================
Feature engineering mirrors the training notebook EXACTLY:
  - Same noise is baked into the trained model
  - No hard binary threshold flags (per notebook design decision)
  - Continuous ratio/interaction features only
  - Audio features are optional; sensor-only mode supported

POST /predict  →  { sensor_data, audio_features, vehicle_info }
GET  /health   →  model status + accuracy
GET  /model-info → full metrics
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Load artefacts ────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent / "models"
model     = joblib.load(BASE / "vehicle_fault_model.pkl")
encoder   = joblib.load(BASE / "label_encoder.pkl")
with open(BASE / "model_metadata.json") as f:
    META = json.load(f)

NUM_FEATURES = META["num_features"]   # 24 columns
CAT_FEATURES = META["cat_features"]   # 7 columns
CLASSES      = META["classes"]

# All valid categorical values (for validation / defaults)
VALID_MAKES     = ["Toyota","Honda","Nissan","Suzuki","Mazda","Hyundai","Kia","Mitsubishi","Perodua","Tata"]
VALID_MODELS    = ["Axio","Premio","Vezel","Lancer","Tiida","Corolla","Rio","Civic","i10","Indica","Fit","Vitz"]
VALID_REGIONS   = ["Hokandara","Battaramulla","Malabe","Nugegoda","Rajagiriya","Maharagama",
                   "Kaduwela","Pannipitiya","Koswatta","Kottawa","Thalawathugoda","Boralesgamuwa",
                   "Athurugiriya","Sri Jayawardenepura Kotte"]
VALID_PATTERNS  = ["mixed","traffic","highway"]
VALID_TRANSMIT  = ["Automatic","Manual"]
VALID_ENGINES   = ["4-cylinder","3-cylinder"]
VALID_FUELS     = ["Petrol","Diesel","Petrol Hybrid"]

app = Flask(__name__)
CORS(app)


# ── Feature engineering — exact mirror of training notebook ──────────────────
def engineer_features(raw: dict) -> pd.DataFrame:
    d = dict(raw)

    # ── Electrical ──
    d["battery_age_stress"]       = d["mileage_km"] / (d["battery_voltage"] + 0.1)
    d["voltage_deviation"]        = abs(d["battery_voltage"] - 12.6)

    # ── Thermal ──
    d["engine_stress_index"]      = d["engine_temp_c"] * d["rpm"] / 1000
    d["temp_mileage_interaction"] = d["engine_temp_c"] * np.log1p(d["mileage_km"])
    d["traffic_heat_load"]        = d["engine_temp_c"] * (1 if d.get("driving_pattern") == "traffic" else 0)
    d["thermal_margin"]           = 115.0 - d["engine_temp_c"]

    # ── Lubrication ──
    d["oil_temp_ratio"]           = d["oil_pressure_psi"] / (d["engine_temp_c"] + 1)
    d["oil_pressure_deviation"]   = abs(d["oil_pressure_psi"] - 40.0)

    # ── Braking ──
    d["brake_wear_index"]         = (100 - d["brake_fluid_level_percent"]) * d["speed_kmh"]
    d["brake_fluid_margin"]       = d["brake_fluid_level_percent"] - 50.0

    # ── Drivetrain ──
    d["rpm_speed_ratio"]          = d["rpm"] / (d["speed_kmh"] + 1)
    d["clutch_slip_index"]        = (d["rpm"] / (d["speed_kmh"] + 1)
                                     if d.get("transmission") == "Manual" else 0.0)

    # ── Vehicle lifecycle ──
    d["vehicle_age"]              = 2025 - int(d.get("year", 2015))
    d["age_mileage_ratio"]        = d["mileage_km"] / (d["vehicle_age"] + 1)
    d["high_mileage_flag"]        = 1 if d["mileage_km"] > 150000 else 0

    # ── Composite health score (continuous 0–100) ──
    d["composite_health_score"] = max(0.0, min(100.0,
        (d["battery_voltage"]           / 14.4) * 25 +
        (d["oil_pressure_psi"]          / 55.0) * 25 +
        (d["brake_fluid_level_percent"] / 95.0) * 25 +
        ((130 - d["engine_temp_c"])     / 55.0) * 25
    ))

    # ── Categorical defaults ──
    d.setdefault("driving_pattern", "mixed")
    d.setdefault("make",            "Toyota")
    d.setdefault("model",           "Axio")
    d.setdefault("engine_type",     "4-cylinder")
    d.setdefault("fuel_type",       "Petrol")
    d.setdefault("transmission",    "Automatic")
    d.setdefault("region_driven",   "Hokandara")

    return pd.DataFrame([d])[NUM_FEATURES + CAT_FEATURES]


def get_recommended_actions(fault: str, sensor: dict) -> list:
    actions = {
        "Healthy": [
            "✅ No faults detected — continue regular maintenance schedule.",
            "📅 Next service due as per manufacturer interval.",
        ],
        "Engine Overheating": [
            "🛑 Stop driving — pull over safely and switch off engine immediately.",
            "🌡️ Allow engine to cool for at least 30 minutes before opening bonnet.",
            "💧 Check coolant level and inspect for leaks.",
            "🔧 Inspect radiator, thermostat, water pump, and cooling fan.",
            "🏥 Do not restart until system inspected by a mechanic.",
        ],
        "Oil Pressure Fault": [
            "⚠️ Reduce speed and pull over — continued driving risks engine seizure.",
            "🛢️ Check engine oil level immediately — top up if low.",
            "🔍 Inspect for oil leaks under the vehicle.",
            "🔧 Check oil pressure sensor and pump.",
            "🚫 Do not drive until oil pressure is confirmed normal.",
        ],
        "Battery Fault": [
            "🔋 Check battery voltage with a multimeter — healthy range 12.4–12.8V.",
            "🔌 Inspect battery terminals for corrosion — clean if needed.",
            "⚡ Test alternator output (should be 13.5–14.5V at idle).",
            "🔁 Battery may need replacement if >4 years old.",
            "💡 Turn off all non-essential electrical loads.",
        ],
        "Brake System Fault": [
            "🛑 Drive to a safe stop immediately — brake failure is a safety emergency.",
            "💧 Check brake fluid level — low level indicates leak or worn pads.",
            "🔊 Listen for grinding/squealing sounds when braking.",
            "🔧 Inspect brake pads, rotors, and brake lines.",
            "🚫 Do not drive until brakes are inspected and repaired.",
        ],
        "Clutch Slip": [
            "⚠️ Avoid heavy acceleration — clutch slipping causes rapid wear.",
            "🔧 Inspect clutch plate, pressure plate, and release bearing.",
            "📊 Check if RPM rises without corresponding speed increase.",
            "🛠️ Clutch adjustment or replacement may be required.",
            "🏥 Have a mechanic inspect the clutch mechanism.",
        ],
    }
    return actions.get(fault, ["🔧 Consult a qualified mechanic for further diagnosis."])


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "model":    META["model_name"],
        "accuracy": META["accuracy"],
        "macro_f1": META["macro_f1"],
        "cv_f1":    f"{META['cv_f1_mean']:.4f} ± {META['cv_f1_std']:.4f}",
        "classes":  CLASSES,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        sensor  = payload.get("sensor_data", {})
        vehicle = payload.get("vehicle_info", {})
        audio   = payload.get("audio_features", {})   # optional

        # Merge all inputs
        raw = {**sensor, **vehicle}

        # Required sensor fields validation
        required = [
            "engine_temp_c", "rpm", "battery_voltage",
            "oil_pressure_psi", "brake_fluid_level_percent",
            "speed_kmh", "mileage_km",
        ]
        missing = [k for k in required if k not in raw or raw[k] is None]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        # Type cast
        for k in required:
            raw[k] = float(raw[k])
        if "year" in raw:
            raw["year"] = int(raw["year"])

        # Build feature row
        X = engineer_features(raw)

        # Predict
        pred_idx   = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        fault      = encoder.inverse_transform([pred_idx])[0]
        confidence = float(pred_proba.max()) * 100

        probs = {cls: round(float(p) * 100, 2)
                 for cls, p in zip(CLASSES, pred_proba)}
        probs_sorted = dict(sorted(probs.items(), key=lambda x: -x[1]))

        # Severity
        severity = (
            "low"      if fault == "Healthy"
            else "critical" if any(k in fault for k in ["Overheating", "Brake", "Oil Pressure"])
            else "moderate"
        )

        health_score = float(X["composite_health_score"].values[0])

        return jsonify({
            "prediction":             fault,
            "confidence":             round(confidence, 1),
            "severity":               severity,
            "probabilities":          probs_sorted,
            "composite_health_score": round(health_score, 1),
            "recommended_actions":    get_recommended_actions(fault, raw),
            "model":                  META["model_name"],
            "model_accuracy":         META["accuracy"],
            "audio_used":             bool(audio),
            "sensor_snapshot": {
                "engine_temp_c":             raw["engine_temp_c"],
                "rpm":                       raw["rpm"],
                "battery_voltage":           raw["battery_voltage"],
                "oil_pressure_psi":          raw["oil_pressure_psi"],
                "brake_fluid_level_percent": raw["brake_fluid_level_percent"],
                "speed_kmh":                 raw["speed_kmh"],
            },
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model":          META["model_name"],
        "accuracy":       META["accuracy"],
        "macro_f1":       META["macro_f1"],
        "cv_f1_mean":     META["cv_f1_mean"],
        "cv_f1_std":      META["cv_f1_std"],
        "cv_scores":      META["cv_scores"],
        "classes":        CLASSES,
        "train_samples":  META["train_samples"],
        "test_samples":   META["test_samples"],
        "per_class":      META["per_class_report"],
        "confusion_matrix": META["confusion_matrix"],
        "feature_note":   "Hard binary threshold flags excluded per pipeline design. Continuous ratio features only.",
    })


@app.route("/demo-cases", methods=["GET"])
def demo_cases():
    """Return example inputs for frontend demo buttons."""
    return jsonify([
        {
            "label": "Toyota Axio — Oil Pressure Fault",
            "sensor_data": {"engine_temp_c": 112.5, "rpm": 2800, "battery_voltage": 12.4,
                            "oil_pressure_psi": 16.0, "brake_fluid_level_percent": 74.0,
                            "speed_kmh": 48, "mileage_km": 145000, "driving_pattern": "mixed"},
            "vehicle_info": {"make": "Toyota", "model": "Axio", "year": 2014,
                             "engine_type": "4-cylinder", "fuel_type": "Petrol",
                             "transmission": "Automatic", "region_driven": "Hokandara"},
        },
        {
            "label": "Honda Civic — Engine Overheating",
            "sensor_data": {"engine_temp_c": 108.0, "rpm": 3200, "battery_voltage": 12.5,
                            "oil_pressure_psi": 31.0, "brake_fluid_level_percent": 72.0,
                            "speed_kmh": 65, "mileage_km": 178000, "driving_pattern": "traffic"},
            "vehicle_info": {"make": "Honda", "model": "Civic", "year": 2012,
                             "engine_type": "4-cylinder", "fuel_type": "Petrol",
                             "transmission": "Automatic", "region_driven": "Battaramulla"},
        },
        {
            "label": "Suzuki Alto — Battery Fault",
            "sensor_data": {"engine_temp_c": 86.0, "rpm": 1400, "battery_voltage": 10.8,
                            "oil_pressure_psi": 37.0, "brake_fluid_level_percent": 76.0,
                            "speed_kmh": 0, "mileage_km": 92000, "driving_pattern": "mixed"},
            "vehicle_info": {"make": "Suzuki", "model": "Axio", "year": 2016,
                             "engine_type": "3-cylinder", "fuel_type": "Petrol",
                             "transmission": "Manual", "region_driven": "Nugegoda"},
        },
        {
            "label": "Toyota Vezel — Healthy",
            "sensor_data": {"engine_temp_c": 86.0, "rpm": 1800, "battery_voltage": 12.7,
                            "oil_pressure_psi": 43.0, "brake_fluid_level_percent": 82.0,
                            "speed_kmh": 42, "mileage_km": 61000, "driving_pattern": "mixed"},
            "vehicle_info": {"make": "Toyota", "model": "Vezel", "year": 2019,
                             "engine_type": "4-cylinder", "fuel_type": "Petrol Hybrid",
                             "transmission": "Automatic", "region_driven": "Malabe"},
        },
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting VehicleAI API on port {port}")
    print(f"Model: {META['model_name']} | Accuracy: {META['accuracy']} | F1: {META['macro_f1']}")
    app.run(host="0.0.0.0", port=port, debug=False)

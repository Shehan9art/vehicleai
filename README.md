# VehicleAI — Engine Fault Detection System

**Final Year Project | University of Wolverhampton | M20010807010**  
N.M. Shehan Dilhara Bandara · Hokandara / Colombo Region

---

## Overview

A production-ready web application for diagnosing four-cylinder vehicle engine faults using combined **sensor data** (OBD-II) and **engine audio** (microphone / file upload).

The system classifies vehicle condition into six categories:

| Class | Description |
|---|---|
| ✅ Healthy | No fault detected |
| 🔥 Engine Overheating | Coolant/thermal system fault |
| 🛢️ Oil Pressure Fault | Oil pump, leak, or low oil level |
| 🔋 Battery Fault | Battery/alternator issue |
| 🛑 Brake System Fault | Pad wear, fluid leak, or rotor damage |
| ⚙️ Clutch Slip | Clutch plate wear (Manual transmission) |

---

## Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | **85.25%** |
| Macro F1 Score | **85.11%** |
| CV F1 (5-fold) | **83.35% ± 1.54%** |
| Algorithm | XGBoost |
| Dataset | 1,217 records |
| Train/Test split | 80/20 stratified |

> **Note:** Accuracy is intentionally realistic (~85%) — not 100%. Hard binary threshold flags were excluded (per pipeline design) to avoid trivial overfitting. The model generalises from continuous sensor ratios and interaction features, consistent with OBD-II sensor noise injection (±1.5°C, ±0.08V, ±1.2psi, etc.).

---

## Project Structure

```
vehicleai/
├── backend/
│   ├── app.py                    # Flask API (predict, health, model-info)
│   ├── audio_extractor.py        # Server-side librosa feature extraction
│   └── requirements.txt
├── frontend/
│   └── index.html                # Complete single-page web application
├── dataset/
│   ├── audio_features.csv        # Generated audio feature dataset (1,217 records)
│   ├── generate_audio_dataset.py # Regenerate audio CSV from sensor/fault CSVs
│   └── data/                     # Place sensor_readings.csv, fault_labels.csv, vehicle_metadata.csv here
├── models/
│   ├── vehicle_fault_model.pkl   # Trained XGBoost pipeline (preprocessor + model)
│   ├── label_encoder.pkl         # LabelEncoder (6 classes)
│   └── model_metadata.json       # Accuracy, F1, CV scores, feature lists, confusion matrix
├── notebooks/
│   └── training_pipeline.ipynb   # Full training notebook (mirrors the ML pipeline)
├── netlify.toml                  # Frontend deployment (Netlify)
├── render.yaml                   # Backend deployment (Render)
├── Procfile                      # Backend deployment (Heroku/Railway)
└── README.md
```

---

## Quickstart — Run Locally

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/vehicleai.git
cd vehicleai
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
# API running on http://localhost:5000
```

### 3. Frontend

```bash
# Simply open in browser:
open frontend/index.html

# OR serve with any static server:
cd frontend
python3 -m http.server 3000
# Visit http://localhost:3000
```

### 4. Test the API

```bash
curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": {
      "engine_temp_c": 112.5,
      "rpm": 2800,
      "battery_voltage": 12.4,
      "oil_pressure_psi": 16.0,
      "brake_fluid_level_percent": 74.0,
      "speed_kmh": 48,
      "mileage_km": 145000,
      "driving_pattern": "mixed"
    },
    "vehicle_info": {
      "make": "Toyota", "model": "Axio", "year": 2014,
      "engine_type": "4-cylinder", "fuel_type": "Petrol",
      "transmission": "Automatic", "region_driven": "Hokandara"
    }
  }' | python3 -m json.tool
```

Expected response:
```json
{
  "prediction": "Oil Pressure Fault",
  "confidence": 88.6,
  "severity": "critical",
  "recommended_actions": ["⚠️ Reduce speed...", "🛢️ Check engine oil..."],
  "composite_health_score": 54.3
}
```

---

## Regenerate the Audio Dataset

```bash
python dataset/generate_audio_dataset.py \
  --sensor_csv dataset/data/sensor_readings.csv \
  --fault_csv  dataset/data/fault_labels.csv \
  --output     dataset/audio_features.csv
```

Audio feature schema: 13 MFCCs (mean + std), spectral centroid, bandwidth, rolloff, ZCR, RMS energy, dominant frequency, tempo BPM — 44 columns total.

Physical signatures per fault:
- **Overheating**: detonation knock at ~2.9 kHz, elevated ZCR
- **Oil Pressure**: hydraulic lifter tapping at ~1.2 kHz
- **Battery Fault**: alternator AC hum at ~150 Hz
- **Brake Fault**: friction squeal at ~3.6 kHz, high spectral rolloff
- **Clutch Slip**: slip rumble at ~30 Hz, erratic tempo

---

## Deploy

### Frontend → Netlify

1. Push to GitHub
2. Connect repo to [Netlify](https://netlify.com)
3. Build command: *(leave empty)*
4. Publish directory: `frontend`
5. Set environment variable `API_URL` in `frontend/index.html` (replace `http://localhost:5000`)

### Backend → Render

1. Create new **Web Service** on [Render](https://render.com)
2. Connect GitHub repo
3. Build command: `pip install -r backend/requirements.txt`
4. Start command: `gunicorn backend.app:app --bind 0.0.0.0:$PORT --workers 2`
5. Upload `models/` folder or build using the training pipeline

---

## ML Pipeline Features

### Sensor Features (24 numeric)
| Feature | Engineering |
|---|---|
| `battery_age_stress` | mileage / voltage |
| `voltage_deviation` | \|voltage − 12.6\| |
| `engine_stress_index` | temp × rpm / 1000 |
| `thermal_margin` | 115 − engine_temp |
| `oil_temp_ratio` | oil_psi / (temp + 1) |
| `clutch_slip_index` | rpm / (speed + 1) if Manual, else 0 |
| `composite_health_score` | weighted 0–100 score |
| + 6 raw sensor values | + 7 interaction/lifecycle features |

> Hard binary threshold flags (`overheat_flag`, `critical_oil_flag`, etc.) were **intentionally excluded** — these create trivial shortcuts that inflate accuracy without learning real sensor relationships.

### Categorical Features (7)
`make`, `model`, `engine_type`, `fuel_type`, `transmission`, `region_driven`, `driving_pattern`

### Noise Injection (realism)
| Sensor | Noise σ |
|---|---|
| engine_temp_c | ±1.5 °C |
| battery_voltage | ±0.08 V |
| oil_pressure_psi | ±1.2 PSI |
| brake_fluid_level | ±2.0 % |
| rpm | ±30 rpm |
| speed_kmh | ±1.5 km/h |

---

## API Reference

### `POST /predict`
```json
{
  "sensor_data": {
    "engine_temp_c": 87.0,    // required, 50–130
    "rpm": 1500,               // required, 0–7000
    "battery_voltage": 12.6,  // required, 9–15
    "oil_pressure_psi": 42.0, // required, 0–80
    "brake_fluid_level_percent": 80.0, // required, 0–100
    "speed_kmh": 40,           // required, 0–200
    "mileage_km": 120000,      // required
    "driving_pattern": "mixed" // optional: mixed|traffic|highway
  },
  "vehicle_info": { ... },     // optional — improves accuracy
  "audio_features": { ... }    // optional — from browser Web Audio API
}
```

### `GET /health` — Model status
### `GET /model-info` — Full metrics, per-class F1, confusion matrix
### `GET /demo-cases` — 4 pre-built test cases

---

## Academic Notes

- Dataset: 1,217 OBD-II sensor readings, Colombo region vehicles (2006–2024)
- Noise injection replicates real OBD-II sensor tolerances (ADC quantisation, EMI, calibration drift)
- Feature engineering rationale documented inline in `backend/app.py` and `notebooks/training_pipeline.ipynb`
- Model comparison: Logistic Regression, KNN, Random Forest, Gradient Boosting, SVM, XGBoost, CatBoost
- Best: XGBoost (n_estimators=800, lr=0.04, max_depth=8, subsample=0.9)
- Evaluation: test accuracy, macro F1, per-class F1, 5-fold CV, confusion matrix, SHAP

---

## Literature References (selected)

- "Hybrid Sensor–Audio Based Fault Detection" (2024)
- "The AI Mechanic: Acoustic Vehicle Characterisation" (2023)
- "CNN Architectures for Audio Classification" (2022)
- "OBD-II based Predictive Maintenance" (2024)
- Full bibliography: see `FP/Lit files/`

---

*University of Wolverhampton · 6CS007 Final Year Project · 2025–2026*

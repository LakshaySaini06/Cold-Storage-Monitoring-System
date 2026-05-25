"""
Cold Storage Monitoring Dashboard - Flask Backend
Run: python server.py
Then open: http://localhost:5000
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import joblib
import pandas as pd
import json
import os
import csv
import threading
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config & model loader ──────────────────────────────────────────────────────

def load_config(storage_type):
    path = os.path.join(BASE_DIR, f"{storage_type}_config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_model(storage_type):
    cfg = load_config(storage_type)
    if not cfg:
        return None
    model_path = os.path.join(BASE_DIR, cfg["model_file"])
    if not os.path.isfile(model_path):
        return None
    return joblib.load(model_path)

# Pre-load models at startup
models = {}
configs = {}
for stype in ("dairy", "pharma"):
    configs[stype] = load_config(stype)
    models[stype]  = load_model(stype)
    if models[stype]:
        print(f"[✓] {stype.upper()} model loaded")
    else:
        print(f"[!] {stype.upper()} model NOT found — run train_{stype}_model.py first")

# ── Sensor data helpers ────────────────────────────────────────────────────────

SENSOR_FILE = os.path.join(BASE_DIR, "sensor_data.csv")

def read_sensor_data(n=50):
    """Read last n rows from sensor_data.csv"""
    if not os.path.isfile(SENSOR_FILE):
        return []
    rows = []
    with open(SENSOR_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "time":        row.get("time") or row.get("timestamp", ""),
                    "temperature": float(row["temperature"]),
                    "humidity":    float(row["humidity"]),
                })
            except (ValueError, KeyError):
                continue
    return rows[-n:]

def rule_check(temp, hum, cfg):
    violations = []
    if temp < cfg["safe_temp_min"]:
        violations.append(f"Temp too LOW ({temp}°C < {cfg['safe_temp_min']}°C)")
    if temp > cfg["safe_temp_max"]:
        violations.append(f"Temp too HIGH ({temp}°C > {cfg['safe_temp_max']}°C)")
    if hum < cfg["safe_hum_min"]:
        violations.append(f"Humidity too LOW ({hum}% < {cfg['safe_hum_min']}%)")
    if hum > cfg["safe_hum_max"]:
        violations.append(f"Humidity too HIGH ({hum}% > {cfg['safe_hum_max']}%)")
    return violations

def run_ml(storage_type, rows):
    """Run ML prediction on a list of {temperature, humidity} dicts"""
    model = models.get(storage_type)
    cfg   = configs.get(storage_type)
    if not model or not cfg or not rows:
        return rows

    df = pd.DataFrame(rows)[["temperature", "humidity"]]
    preds = model.predict(df)
    probas = model.predict_proba(df)[:, 1]

    for i, row in enumerate(rows):
        row["ml_prediction"] = int(preds[i])
        row["ml_label"]      = "ABNORMAL" if preds[i] == 1 else "normal"
        row["ml_confidence"] = round(float(probas[i]) * 100, 1)
        row["violations"]    = rule_check(row["temperature"], row["humidity"], cfg)
    return rows

# ── API Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    """Return connected models and config info"""
    result = {}
    for stype in ("dairy", "pharma"):
        cfg = configs.get(stype)
        result[stype] = {
            "model_loaded": models.get(stype) is not None,
            "config":       cfg,
        }
    return jsonify(result)

@app.route("/api/readings")
def api_readings():
    """Latest sensor readings with ML predictions"""
    storage_type = request.args.get("type", "dairy")
    n            = int(request.args.get("n", 50))

    rows = read_sensor_data(n)
    if not rows:
        return jsonify({"error": "No sensor data found. Make sure sensor_data.csv exists.", "rows": []})

    rows = run_ml(storage_type, rows)
    cfg  = configs.get(storage_type, {})

    # Summary
    alert_window    = cfg.get("alert_window", 5) if cfg else 5
    alert_threshold = cfg.get("alert_threshold", 3) if cfg else 3
    recent          = rows[-alert_window:]
    abnormal_count  = sum(1 for r in recent if r.get("ml_prediction") == 1)
    rule_violations = sum(1 for r in recent if r.get("violations"))

    latest = rows[-1] if rows else {}
    system_alert = (abnormal_count >= alert_threshold) or (rule_violations > 0)

    return jsonify({
        "rows":             rows,
        "latest":           latest,
        "summary": {
            "total":           len(rows),
            "abnormal":        sum(1 for r in rows if r.get("ml_prediction") == 1),
            "normal":          sum(1 for r in rows if r.get("ml_prediction") == 0),
            "recent_abnormal": abnormal_count,
            "rule_violations": rule_violations,
            "system_alert":    system_alert,
            "alert_window":    alert_window,
            "alert_threshold": alert_threshold,
            "checked_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    })

@app.route("/api/config/<storage_type>")
def api_config(storage_type):
    cfg = configs.get(storage_type)
    if not cfg:
        return jsonify({"error": f"Config for '{storage_type}' not found"}), 404
    return jsonify(cfg)

@app.route("/api/reload_models", methods=["POST"])
def reload_models():
    """Hot-reload models without restarting server"""
    for stype in ("dairy", "pharma"):
        configs[stype] = load_config(stype)
        models[stype]  = load_model(stype)
    return jsonify({"status": "reloaded", "dairy": models["dairy"] is not None, "pharma": models["pharma"] is not None})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  COLD STORAGE DASHBOARD SERVER")
    print("="*50)
    print(f"  Open browser: http://localhost:5000")
    print(f"  Working dir : {BASE_DIR}")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
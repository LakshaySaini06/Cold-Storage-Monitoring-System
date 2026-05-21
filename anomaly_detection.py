import joblib
import pandas as pd
import json
import os
from datetime import datetime

print("  COLD STORAGE ANOMALY DETECTION  ")

storage_type = input("Enter storage type (pharma / dairy): ").strip().lower()
 
if storage_type not in ('pharma', 'dairy'):
    print("[ERROR] Invalid storage type. Choose 'pharma' or 'dairy'.")
    exit(1)
 
config_file = f"{storage_type}_config.json"
 
if not os.path.isfile(config_file):
    print(f"[ERROR] Config file '{config_file}' not found.")
    print(f"        Please run 'train_{storage_type}_model.py' first.")
    exit(1)
 
with open(config_file) as f:
    config = json.load(f)
 
SAFE_TEMP_MIN  = config['safe_temp_min']
SAFE_TEMP_MAX  = config['safe_temp_max']
SAFE_HUM_MIN   = config['safe_hum_min']
SAFE_HUM_MAX   = config['safe_hum_max']
ALERT_WINDOW   = config['alert_window']
ALERT_THRESHOLD= config['alert_threshold']
FEATURES       = config['features']
 
print(f"\n[INFO] Storage type : {storage_type.upper()}")
print(f"[INFO] Safe temp    : {SAFE_TEMP_MIN}–{SAFE_TEMP_MAX} °C")
print(f"[INFO] Safe humidity: {SAFE_HUM_MIN}–{SAFE_HUM_MAX} %")
 

model_file = config['model_file']
 
if not os.path.isfile(model_file):
    print(f"[ERROR] Model file '{model_file}' not found.")
    print(f"        Please run 'train_{storage_type}_model.py' first.")
    exit(1)
 
model = joblib.load(model_file)
print(f"[INFO] Model loaded : {model_file}")
 

sensor_file = 'sensor_data.csv'
 
if not os.path.isfile(sensor_file):
    print(f"[ERROR] Sensor data file '{sensor_file}' not found.")
    exit(1)
 
df = pd.read_csv(sensor_file)
 
required_cols = {'temperature', 'humidity'}
if not required_cols.issubset(df.columns):
    print(f"[ERROR] sensor_data.csv must have columns: {required_cols}")
    exit(1)
 
df = df.dropna(subset=['temperature', 'humidity'])
 
if df.empty:
    print("[ERROR] No valid sensor readings found in sensor_data.csv.")
    exit(1)

recent = df.tail(ALERT_WINDOW).copy()
print(f"\n[INFO] Analysing last {len(recent)} reading(s)...")
 

def rule_check(row):
    """Returns a string describing any rule violation, or None if OK."""
    violations = []
    if row['temperature'] < SAFE_TEMP_MIN:
        violations.append(f"Temp too LOW ({row['temperature']}°C < {SAFE_TEMP_MIN}°C)")
    if row['temperature'] > SAFE_TEMP_MAX:
        violations.append(f"Temp too HIGH ({row['temperature']}°C > {SAFE_TEMP_MAX}°C)")
    if row['humidity'] < SAFE_HUM_MIN:
        violations.append(f"Humidity too LOW ({row['humidity']}% < {SAFE_HUM_MIN}%)")
    if row['humidity'] > SAFE_HUM_MAX:
        violations.append(f"Humidity too HIGH ({row['humidity']}% > {SAFE_HUM_MAX}%)")
    return violations

X_recent = recent[FEATURES]
predictions = model.predict(X_recent)       
probabilities = model.predict_proba(X_recent)[:, 1]   
 
recent = recent.copy()
recent['ml_prediction'] = predictions
recent['ml_confidence'] = probabilities.round(3)
 

print("\n" + "─"*60)
print(f"  RECENT READINGS — {storage_type.upper()} STORAGE")
print("─"*60)
 
has_time_col = 'time' in recent.columns or 'timestamp' in recent.columns
time_col = 'timestamp' if 'timestamp' in recent.columns else ('time' if 'time' in recent.columns else None)
 
for idx, row in recent.iterrows():
    status = "⚠  ABNORMAL" if row['ml_prediction'] == 1 else "✓  normal  "
    time_str = f" [{row[time_col]}]" if time_col else ""
    violations = rule_check(row)
    rule_str = " | RULE: " + "; ".join(violations) if violations else ""
    print(f"  {status} | Temp: {row['temperature']:5.1f}°C | Hum: {row['humidity']:5.1f}% "
          f"| Confidence: {row['ml_confidence']:.0%}{time_str}{rule_str}")
 

abnormal_count = int(predictions.sum())
rule_violations = sum(1 for _, row in recent.iterrows() if rule_check(row))
 
print("\n" + "─"*60)
print(f"  SUMMARY")
print("─"*60)
print(f"  Readings analysed : {len(recent)}")
print(f"  ML flagged        : {abnormal_count} / {len(recent)}")
print(f"  Rule violations   : {rule_violations} / {len(recent)}")
 
print()
 

if abnormal_count >= ALERT_THRESHOLD or rule_violations > 0:
  
    print("    ALERT: ABNORMAL CONDITION DETECTED    ")
    
    if abnormal_count >= ALERT_THRESHOLD:
        print(f"\n  ML model flagged {abnormal_count}/{len(recent)} recent readings as abnormal.")
    if rule_violations > 0:
        print(f"\n  {rule_violations} reading(s) breached safe {storage_type} thresholds.")
    print(f"\n   Action: Check {storage_type.upper()} storage unit immediately.")
    print(f"   Safe range: {SAFE_TEMP_MIN}–{SAFE_TEMP_MAX}°C | {SAFE_HUM_MIN}–{SAFE_HUM_MAX}% RH")
else:
    avg_temp = recent['temperature'].mean()
    avg_hum  = recent['humidity'].mean()
    print(f"    All readings NORMAL")
    print(f"     Avg temp: {avg_temp:.1f}°C | Avg humidity: {avg_hum:.1f}%")
 
print(f"\n  Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("─"*60 + "\n")
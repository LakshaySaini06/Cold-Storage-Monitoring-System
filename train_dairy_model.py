import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import joblib
import json
import os

STORAGE_TYPE = "dairy"

DATA_FILE = "dairy_training_dataset.csv"
MODEL_FILE = "dairy_model.pkl"
CONFIG_FILE = "dairy_config.json"

SAFE_TEMP_MIN = 1.0
SAFE_TEMP_MAX = 4.0

SAFE_HUM_MIN = 75.0
SAFE_HUM_MAX = 90.0

print("  DAIRY MODEL TRAINING")

if not os.path.isfile(DATA_FILE):
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

df = df.dropna()

required_columns = ['temperature', 'humidity', 'condition']

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

print(f"\n[INFO] Loaded {len(df)} rows from '{DATA_FILE}'")

print(f"\n[INFO] Label distribution:")
print(df['condition'].value_counts().to_string())


X = df[['temperature', 'humidity']]

y = df['condition'].map({
    'normal': 0,
    'abnormal': 1
})


if y.isnull().any():
    raise ValueError(
        "Unknown label values found. "
        "Expected 'normal' or 'abnormal'."
    )


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n[INFO] Train samples : {len(X_train)}")
print(f"[INFO] Test samples  : {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

print("\n[INFO] Model trained successfully.")

y_pred = model.predict(X_test)

print("\n" + "─"*50)
print("  CLASSIFICATION REPORT")
print("─"*50)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=['normal', 'abnormal']
    )
)

accuracy = accuracy_score(y_test, y_pred)

print(f"[INFO] Accuracy : {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(f"  True Normal  (TN): {cm[0][0]}")
print(f"  False Alarm  (FP): {cm[0][1]}")
print(f"  Missed Alert (FN): {cm[1][0]}")
print(f"  True Abnormal(TP): {cm[1][1]}")


cv_folds = min(5, len(df) // 2)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv_folds,
    scoring='f1'
)

print(f"\n[INFO] Cross-val F1 scores : {cv_scores.round(3)}")
print(f"[INFO] Mean CV F1          : {cv_scores.mean():.3f}")
print(f"[INFO] Std Dev             : {cv_scores.std():.3f}")


feat_imp = dict(
    zip(
        X.columns,
        model.feature_importances_.round(4)
    )
)

print(f"\n[INFO] Feature importances:")
print(feat_imp)


joblib.dump(model, MODEL_FILE)

print(f"\n[✓] Model saved → '{MODEL_FILE}'")


config = {
    "storage_type": STORAGE_TYPE,
    "model_file": MODEL_FILE,

    "safe_temp_min": SAFE_TEMP_MIN,
    "safe_temp_max": SAFE_TEMP_MAX,

    "safe_hum_min": SAFE_HUM_MIN,
    "safe_hum_max": SAFE_HUM_MAX,

    "alert_window": 5,
    "alert_threshold": 3,

    "features": list(X.columns)
}

with open(CONFIG_FILE, 'w') as f:
    json.dump(config, f, indent=4)

print(f"[✓] Config saved → '{CONFIG_FILE}'")

print(f"\n{'='*50}")
print("  Dairy model ready.")
print(
    f"  Safe range: "
    f"{SAFE_TEMP_MIN}-{SAFE_TEMP_MAX}°C | "
    f"{SAFE_HUM_MIN}-{SAFE_HUM_MAX}% RH"
)

"""Train the leak-resistant, calibrated placement-probability model.

Run from the project root:
    python models/train_placement_model.py
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from feature_engineering import ENGINEERED_FEATURES, MODEL_FEATURES, add_placement_features

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "student_placement_feature_engineered.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "placement_prediction_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "placement_model_metrics.json"

# Only use information available before a placement outcome is known.
FEATURES = MODEL_FEATURES
CATEGORICAL_FEATURES = ["gender", "degree", "branch"]
NUMERICAL_FEATURES = [column for column in FEATURES if column not in CATEGORICAL_FEATURES]


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    # Recompute in the shared function so the processed file and deployment formula cannot drift.
    data = add_placement_features(data)
    required = set(FEATURES + ["placed"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    X = data[FEATURES].copy()
    y = pd.to_numeric(data["placed"], errors="raise").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", CalibratedClassifierCV(forest, method="sigmoid", cv=5, n_jobs=-1)),
        ]
    )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
        "feature_columns": FEATURES,
        "training_dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "engineered_features": ENGINEERED_FEATURES,
        "note": "Metrics use a held-out 20% test set. Probabilities are estimates, not guarantees.",
    }
    print(json.dumps(metrics, indent=2))

    # Refit after evaluation so the saved model can use all available labelled records.
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved calibrated model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from model_utils import build_model, FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "titanic_model.joblib"
DATA_PATH = ROOT / "data" / "train.csv"

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-for-production"


def load_or_train_model():
    """Load a compatible local model or train a lightweight model from train.csv."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    if DATA_PATH.exists():
        train = pd.read_csv(DATA_PATH)
    else:
        # Download the official Kaggle dataset and place train.csv in data/.
        # This fallback keeps the demo runnable without a local dataset.
        train = pd.DataFrame({
            "Survived": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "Pclass": [3, 1, 3, 2, 1, 3, 2, 3, 1, 3, 2, 3],
            "Sex": ["male", "female", "female", "male", "female", "male", "female", "male", "female", "male", "female", "male"],
            "Age": [22, 38, 26, 35, 28, 30, 18, 40, 29, 25, 32, 45],
            "SibSp": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "Parch": [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            "Fare": [7.25, 71.28, 7.92, 13.0, 80.0, 8.05, 26.0, 7.9, 90.0, 7.8, 15.0, 8.0],
            "Embarked": ["S", "C", "S", "S", "C", "S", "Q", "S", "C", "S", "S", "S"],
        })
    model = build_model()
    model.fit(train[FEATURES], train["Survived"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


model = load_or_train_model()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    passenger = {}
    error = None

    if request.method == "POST":
        try:
            passenger = {
                "Pclass": int(request.form.get("pclass", 3)),
                "Sex": request.form.get("sex", "male"),
                "Age": float(request.form.get("age", 28)),
                "SibSp": int(request.form.get("sibsp", 0)),
                "Parch": int(request.form.get("parch", 0)),
                "Fare": float(request.form.get("fare", 14.0)),
                "Embarked": request.form.get("embarked", "S"),
            }
            if not 0 <= passenger["Age"] <= 100:
                raise ValueError("Age must be between 0 and 100.")
            if passenger["Fare"] < 0:
                raise ValueError("Fare cannot be negative.")
            if passenger["SibSp"] < 0 or passenger["Parch"] < 0:
                raise ValueError("Family counts cannot be negative.")

            row = pd.DataFrame([passenger], columns=FEATURES)
            prediction = int(model.predict(row)[0])
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(row)[0][1])
        except (ValueError, TypeError) as exc:
            error = str(exc) or "Please check the passenger details."

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        passenger=passenger,
        error=error,
    )


@app.route("/health")
def health():
    return {"status": "ok", "model": "loaded"}


if __name__ == "__main__":
    app.run(debug=True)

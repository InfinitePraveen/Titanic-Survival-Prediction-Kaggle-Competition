# Titanic Survival Prediction — Flask Demo

A creative, responsive Flask web app for exploring a machine-learning prediction of Titanic passenger survival. Built as an add-on branch for [Titanic-Survival-Prediction-Kaggle-Competition](https://github.com/InfinitePraveen/Titanic-Survival-Prediction-Kaggle-Competition).

> **Important:** This is an educational model demonstration, not a historically reliable reconstruction or a real-world risk assessment. The prediction is probabilistic and can be wrong.

## Features
- Responsive “Titanic passenger manifest” interface with a maritime visual theme.
- Predicts survival from passenger class, sex, age, family aboard, fare, and port of embarkation.
- Displays model probability when available.
- Lightweight scikit-learn pipeline with missing-value handling and categorical encoding.
- Trains automatically from `data/train.csv` if present; otherwise uses a tiny illustrative fallback dataset so the UI can run immediately.
- Health endpoint at `/health`.

## Run locally (Windows)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download `train.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) and put it in `data/train.csv` for a meaningful model. Then run:
```powershell
python app.py
```
Open http://127.0.0.1:5000.

The app trains a local model on first run and saves it to `artifacts/titanic_model.joblib`. Delete that file after replacing the dataset to retrain.

## Add this to a new Git branch
From the root of your existing repository:
```bash
git switch -c feature/flask-titanic-webapp
```
Unzip/copy these files into the repository root. Commit the new app files:
```bash
git add app.py model_utils.py requirements.txt README.md .gitignore templates static data artifacts
git commit -m "Add creative Flask Titanic survival prediction web app"
git push -u origin feature/flask-titanic-webapp
```
If your repository already has a `README.md`, merge the relevant section rather than overwriting your project README.

## Author
**Praveen Kumar**  
GitHub: https://github.com/InfinitePraveen  
LinkedIn: https://www.linkedin.com/in/infinitepraveen/

## Dataset
Kaggle: https://www.kaggle.com/competitions/titanic

# 🚢 Titanic Survival Prediction — Kaggle Competition

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/code/infinitepraveen/titanic-survival-predictions-ipynb?scriptVersionId=330090073)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains my end-to-end solution for the classic Kaggle beginner competition: **Titanic: Machine Learning from Disaster**. The goal of this project is to build a predictive model that answers the question: *"What sorts of people were more likely to survive?"* using passenger data (e.g., name, age, gender, socio-economic class, etc.).

---

## 🔍 Live & Upvotes on Kaggle
- **Live Views**: 45
- **Upvotes**: (Coming soon)
- **Kaggle Notebook Link**: [Click here to view my submission](https://www.kaggle.com/code/infinitepraveen/titanic-survival-predictions-ipynb?scriptVersionId=330090073)

---

## 📁 Project Structure
```
Titanic-Survival-Prediction-Kaggle-Competition/
├── titanic-survival-prediction.ipynb  # Main Jupyter Notebook with full pipeline
├── README.md                          # Project overview and documentation
├── requirements.txt                   # Python dependencies
├── submission.csv                     # Final predictions for Kaggle submission
└── models/                            # Saved trained models (.pkl files)
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    ├── gradient_boosting.pkl
    ├── svm.pkl
    └── decision_tree_pruned.pkl
```

---

## 🧠 Project Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed survival rates by **Sex**, **Pclass**, and **Embarked** ports.
- Visualized **Age** and **Fare** distributions and their relationship with survival.
- Identified missing data patterns and key correlations using a heatmap.

### 2. Data Preprocessing
- **Missing Value Imputation**:
  - `Age`: Imputed with median, stratified by `Pclass` and `Sex`.
  - `Fare`: Imputed with median.
  - `Embarked`: Filled with the mode ('S').
  - `Cabin`: Dropped, but created a `HasCabin` binary feature to capture its predictive signal.
- **Feature Engineering**:
  - **Title**: Extracted from `Name` (e.g., Mr, Mrs, Miss, Master) and grouped rare titles.
  - **FamilySize**: Derived from `SibSp` and `Parch` to capture family dynamics.
  - **IsAlone**: Binary flag for solo passengers.
  - **AgeBand** & **FareBand**: Binned continuous features into meaningful ordinal categories.
- **Encoding & Scaling**:
  - Ordinal mapping for `Sex`, `Embarked`, `Title`, `AgeBand`, and `FareBand`.
  - `FamilySize` was scaled using `StandardScaler`.

### 3. Modeling & Tuning
Five different classifiers were trained and evaluated:

- **Logistic Regression**
- **Decision Tree** (with and without pruning)
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**

**Performance Summary** (Validation Set)

| Model                   | Train Accuracy | Val Accuracy | Overfit Gap |
|-------------------------|----------------|--------------|-------------|
| SVM                     | 83.43%         | **83.24%**   | +0.19%      |
| Logistic Regression     | 81.18%         | 81.56%       | -0.38%      |
| Gradient Boosting       | 86.94%         | 79.89%       | +7.05%      |
| Decision Tree (pruned)  | 85.11%         | 79.33%       | +5.78%      |
| Random Forest           | 90.59%         | 79.33%       | +11.26%     |

**Hyperparameter Tuning** was performed using `GridSearchCV` with 5-fold cross-validation to optimize the best models.

### 4. Evaluation
The final model (Gradient Boosting) was evaluated using:
- **Confusion Matrix**: To visualize correct vs. incorrect predictions.
- **ROC-AUC Curve**: Score of **0.85**, indicating good discriminative power.
- **Feature Importance**: Identified `Sex`, `Title`, `Pclass`, and `FamilySize` as the most influential features.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Jupyter Notebook or Google Colab / Kaggle Notebook

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/InfinitePraveen/Titanic-Survival-Prediction-Kaggle-Competition.git
    cd Titanic-Survival-Prediction-Kaggle-Competition
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebook
Open `titanic-survival-prediction.ipynb` in Jupyter Notebook or any compatible environment and run all cells. The notebook will:
1.  Load the training and test datasets.
2.  Perform EDA.
3.  Preprocess the data and engineer features.
4.  Train, evaluate, and tune multiple models.
5.  Generate a `submission.csv` file ready for Kaggle.

---

## 📈 Results & Submission
The final tuned **Gradient Boosting** model was used to make predictions on the test set. The resulting `submission.csv` file contains the passenger IDs and predicted survival outcomes (0 or 1). The predicted survival rate for the test set was approximately 34.7%.

---

## 📚 Key Learnings
- **Feature engineering** is crucial for improving model performance. Creating features like `Title` and `IsAlone` provided strong predictive signals.
- **Handling missing data** thoughtfully (e.g., group-wise imputation) helps retain valuable information.
- **Pruning** and **regularization** are important to reduce overfitting, especially with tree-based models.
- **Ensemble methods** like Gradient Boosting and Random Forest can achieve high performance but require careful tuning.

---

## 🛠️ Built With
- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning models, preprocessing, and evaluation.
- **Joblib**: Model serialization.

---

## 📄 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author
**Praveen Kumar**
- GitHub: [@InfinitePraveen](https://github.com/InfinitePraveen)
- Kaggle: [@infinitepraveen](https://www.kaggle.com/infinitepraveen)

---

## 🙏 Acknowledgements
- The **Titanic** dataset is provided by Kaggle as part of the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition.
- Inspiration from the Kaggle community and numerous online tutorials on machine learning pipelines.

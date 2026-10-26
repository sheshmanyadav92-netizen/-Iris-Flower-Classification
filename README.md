# Task 1: Iris Flower Classification

## Overview
This repository contains a complete, robust machine learning workflow to classify Iris flower species (*Setosa*, *Versicolor*, *Virginica*) based on morphological measurements of sepal and petal dimensions. 

Built with industry-standard machine learning practices:
- **Zero Data Leakage:** Preprocessing and scaling (`StandardScaler`) are encapsulated strictly within scikit-learn `Pipeline` objects.
- **Deterministic Reproducibility:** Global `random_state=42` applied across all splits, cross-validation folds, and stochastic estimators.
- **Thorough Evaluation:** Held-out test performance (stratified 80/20) benchmarked alongside 5-fold stratified cross-validation on training data.

---

## Directory Structure
```text
Task-1-Iris-Classification/
│   ├── iris_classification.ipynb   # Executed Jupyter Notebook with EDA, pipelines, and evaluation
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # Project documentation and summary
```

---

## Dataset Description
The dataset is loaded directly from `sklearn.datasets.load_iris`. It consists of 150 instances evenly balanced across 3 species (50 instances per class):
- **Features (continuous, cm):**
  - `sepal length (cm)`
  - `sepal width (cm)`
  - `petal length (cm)`
  - `petal width (cm)`
- **Target:** `species` (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*)

---

## Methodology & Pipeline
1. **Setup & Reproducibility:** Fixed random seed (`RANDOM_STATE = 42`) and modern styling configuration.
2. **Data Ingestion:** Loaded via `load_iris(as_frame=True)` into a pandas DataFrame with human-readable target labels.
3. **Exploratory Data Analysis (EDA):** Verification of shapes, data types, complete absence of null values, descriptive statistics, and class balance.
4. **Visualizations:**
   - Pairwise scatter/KDE plots (`pairplot`) colored by species.
   - 2x2 grid of boxplots showing distributions and outliers across species.
   - Pearson correlation heatmap highlighting feature interdependencies.
5. **Feature Importance Analysis:** Quantitative verification using Random Forest split importances:
   - Petal measurements (`petal length` and `petal width`) dominate with ~85-90% of total importance.
   - Sepal features exhibit significant overlap between *Versicolor* and *Virginica*.
6. **Leakage-Free Splitting:** Stratified 80/20 train-test split (`stratify=y`, 120 train / 30 test samples).
7. **Model Architectures:**
   - **Logistic Regression:** Scaled via `Pipeline(StandardScaler -> LogisticRegression)`
   - **K-Nearest Neighbors (KNN):** Scaled via `Pipeline(StandardScaler -> KNeighborsClassifier)`
   - **Decision Tree:** Unscaled CART classifier (`DecisionTreeClassifier`)
   - **Random Forest:** Ensemble of 100 decision trees (`RandomForestClassifier`)
8. **Evaluation:** Detailed classification reports (precision, recall, F1-score), annotated confusion matrix heatmaps, and 5-fold stratified cross-validation.

---

## Experimental Results

| Model | Preprocessing | Test Accuracy | 5-Fold CV Accuracy (Mean ± Std) |
| :--- | :--- | :---: | :---: |
| **Logistic Regression** | `StandardScaler` (Pipeline) | **93.33%** | **95.83% ± 2.64%** |
| **K-Nearest Neighbors (KNN)** | `StandardScaler` (Pipeline) | **93.33%** | **95.83% ± 2.64%** |
| **Decision Tree** | None | 93.33% | 95.00% ± 1.67% |
| **Random Forest** | None | 90.00% | 95.00% ± 3.12% |

### Key Findings & Best Model:
- **Best Model:** **Logistic Regression** (Pipeline with `StandardScaler`) ties KNN for the highest CV accuracy (**95.83%**) and test accuracy (**93.33%**). It is selected as the primary production model due to its linear interpretability, well-calibrated class probabilities, and lightweight inference overhead.
- **Dataset Size Context:** The test set comprises 30 samples (10 per class). A difference of a single sample represents a 3.33% accuracy swing. Therefore, the performance differences between these models are not statistically significant.

---

## Getting Started

### Prerequisites
- Python 3.10+ (tested on Python 3.12)

### Installation
Clone or navigate to the repository directory and install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook iris_classification.ipynb
```
Or open and execute `iris_classification.ipynb` directly in VS Code / Cursor with the Python & Jupyter extensions.

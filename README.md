# Rainfall Prediction Classifier — Melbourne Area

Final project for the IBM Machine Learning course (Skills Network): a binary classifier that predicts whether it will rain **today** in the Melbourne area, based on weather observations from the previous day.

## Problem Statement

The original dataset contains the target `RainTomorrow`. Using it directly would cause **data leakage** in practice: features such as `Rainfall`, `Evaporation` or `Sunshine` are only known after the whole day has passed, so they cannot be used to forecast "tomorrow" in real time.

To make the task realistic, the target is reframed:

| Original column | Renamed to      | Role    |
|-----------------|-----------------|---------|
| `RainToday`     | `RainYesterday` | Feature |
| `RainTomorrow`  | `RainToday`     | Target  |

All other measurements are therefore interpreted as **yesterday's** weather, and the model answers a practical question: *"Should I bike to work today?"*

## Dataset

- **Source:** Australian Bureau of Meteorology, via [Kaggle — Rain in Australia (weatherAUS)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- **Period:** 2008–2017, daily observations
- **Raw size:** 145,460 rows × 23 columns
- **After `dropna()`:** 56,420 rows
- **After location filtering:** 7,557 rows

### Location selection

Weather patterns vary strongly across Australia, so the scope is limited to three nearby stations:

| Location         | Rows  |
|------------------|-------|
| MelbourneAirport | 2,929 |
| Watsonia         | 2,730 |
| Melbourne        | 1,898 |

`Location` is kept as a categorical feature to capture small local differences.

### Class balance

| Class | Share  |
|-------|--------|
| No    | ~76%   |
| Yes   | ~24%   |

The dataset is **moderately imbalanced (~3:1)**. A naive "never rains" baseline already reaches ~77% accuracy, so accuracy alone is not a reliable metric.

## Feature Engineering

- **`Season`** — derived from `Date` using Southern Hemisphere seasons (Dec–Feb = Summer, Mar–May = Autumn, Jun–Aug = Winter, Sep–Nov = Spring). `Date` is dropped afterwards.
- **Numeric features (16):** temperatures, rainfall, evaporation, sunshine, wind speeds, humidity, pressure, cloud cover.
- **Categorical features (6):** `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainYesterday`, `Season`.

After one-hot encoding the model sees **73 features**.

## Modelling Pipeline

```
Pipeline
├── preprocessor: ColumnTransformer
│   ├── num → StandardScaler
│   └── cat → OneHotEncoder(handle_unknown='ignore')
└── classifier: RandomForestClassifier | LogisticRegression
```

- **Train/test split:** 80/20, `random_state=42`
- **Cross-validation:** `StratifiedKFold(n_splits=5, shuffle=True)`
- **Hyperparameter search:** `GridSearchCV`, `scoring='accuracy'`

### Search grids

**Random Forest**

| Parameter           | Values          |
|---------------------|-----------------|
| `n_estimators`      | 50, 100         |
| `max_depth`         | None, 10, 20    |
| `min_samples_split` | 2, 5            |

Best: `n_estimators=100`, `max_depth=None`, `min_samples_split=2` (CV accuracy 0.85).

**Logistic Regression**

| Parameter      | Values            |
|----------------|-------------------|
| `solver`       | liblinear         |
| `penalty`      | l1, l2            |
| `class_weight` | None, balanced    |

## Results (test set, 1,512 samples)

| Metric                  | Random Forest | Logistic Regression |
|-------------------------|:-------------:|:-------------------:|
| Accuracy                | **0.85**      | 0.84                |
| Precision (Yes)         | **0.74**      | 0.69                |
| Recall / TPR (Yes)      | **0.53**      | 0.51                |
| F1-score (Yes)          | **0.62**      | 0.59                |
| Macro-avg recall        | 0.74          | 0.72                |

**Random Forest confusion matrix**

|                | Predicted No | Predicted Yes |
|----------------|:------------:|:-------------:|
| **Actual No**  | 1106         | 62            |
| **Actual Yes** | 161          | 183           |

### Interpretation

- Both models beat the ~77% baseline, with Random Forest slightly ahead on every metric.
- Both are **conservative**: when they predict rain they are right about 3 times out of 4, but they **miss roughly half of the rainy days** (TPR ≈ 0.51–0.53).
- As a rainfall predictor the models are therefore only moderately useful. The main limiting factor is the class imbalance combined with optimising for accuracy.

Feature importances for the Random Forest are plotted in the notebook (Exercise 14).

## Possible Improvements

- Optimise for `f1`, `recall` or `roc_auc` instead of `accuracy` in `GridSearchCV`.
- Use `class_weight='balanced'` for Random Forest as well.
- Tune the decision threshold via `predict_proba` (e.g. 0.30–0.35) to trade precision for recall.
- Add `stratify=y` to `train_test_split` and a fixed `random_state` to `StratifiedKFold` for reproducibility.
- Impute missing values instead of dropping ~60% of rows, to make use of more data.
- Keep `Month` as a feature alongside `Season` for finer seasonality.
- Try gradient boosting models (e.g. `HistGradientBoostingClassifier`).

## Environment

| Package       | Version |
|---------------|---------|
| Python        | 3.14    |
| numpy         | 2.5.4   |
| pandas        | 3.0.6   |
| scikit-learn  | 1.9.1   |
| matplotlib    | 3.11.2  |
| seaborn       | 0.13.2  |

### Compatibility notes

- **pandas 3:** string columns use the new `str` dtype. Use `select_dtypes(include=['object', 'str', 'category'])` to avoid a deprecation warning.
- **scikit-learn ≥ 1.8:** `penalty` in `LogisticRegression` is deprecated. Use `l1_ratio` instead (`0` = L2, `1` = L1).
- **`DataFrame.drop`:** do not pass both `columns=` and `axis=` — this raises a `ValueError`.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook FinalProject_AUSWeather.ipynb
```

Place `weatherAUS_2.csv` next to the notebook before running it.

## Author

Yaroslav Abramiuk — Software Engineering student, Prykarpattian National University.

Original lab: Jeff Grossman, Abhishek Gagneja (IBM Skills Network).

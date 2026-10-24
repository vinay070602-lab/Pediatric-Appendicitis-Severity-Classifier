# Pediatric Appendicitis Severity Prediction Using XGBoost

Machine learning pipeline for classifying **complicated** versus **uncomplicated** pediatric appendicitis using tabular clinical data from the [Regensburg Pediatric Appendicitis Dataset](https://zenodo.org/records/7669215). The repository contains the training and evaluation notebook, hyperparameter optimization, class-imbalance handling, and SHAP-based model interpretation.

> **Research use only.** This experimental model has not been externally validated or approved for clinical decision-making. It must not be used to diagnose patients or guide treatment.

## Overview

Appendicitis severity assessment can inform clinical research into disease progression and risk stratification. This project investigates whether an Extreme Gradient Boosting (XGBoost) classifier can distinguish complicated from uncomplicated cases based on tabular patient features.

The notebook implements:

- Median imputation for numerical features and most-frequent imputation for categorical features
- One-hot encoding of categorical features, ignoring previously unseen categories
- Synthetic Minority Oversampling Technique (SMOTE) within the training pipeline
- XGBoost binary classification and randomized hyperparameter search
- Stratified train/test split and five-fold stratified cross-validation
- Discrimination, classification, and probability-quality metrics
- SHAP visualizations to explore feature contributions

## Dataset

**Source:** [Regensburg Pediatric Appendicitis Dataset — Zenodo record 7669215](https://zenodo.org/records/7669215)

The notebook loads a local, prepared Excel file named `DATA_CLEAN.xlsx`. This is a **preprocessed local input**, not a file automatically downloaded by the notebook. The steps that created `DATA_CLEAN.xlsx` from the original dataset are not documented in this notebook, so users must prepare a compatible file separately.

The notebook's saved output reports **779 labeled records**:

| Severity label | Encoding | Count | Percentage |
|---|---:|---:|---:|
| Uncomplicated | 0 | 660 | 84.7% |
| Complicated | 1 | 119 | 15.3% |
| **Total** | | **779** | **100%** |

`Severity` is the prediction target. Both `Severity` and `Diagnosis` are excluded from the feature matrix. Columns with at least 90% missing values are removed before modeling.

**Important:** The source dataset and local cleaned spreadsheet are not necessarily identical. The counts above describe the data used in the saved notebook run, not necessarily all records in the original Zenodo archive. Ensure your data usage and redistribution follow the source repository's terms.

## Repository Structure

```text
.
├── XGB_severity_model.ipynb   # Training, tuning, evaluation, SHAP analysis
├── DATA_CLEAN.xlsx            # User-supplied local input (not bundled)
└── README.md                  # Project documentation
```

The structure above shows the required local arrangement, not a claim that the dataset is committed to the repository.

## Requirements

- Python 3.10+ recommended (exact version used for the saved notebook run is not documented)
- Jupyter Notebook or JupyterLab
- NumPy, pandas, matplotlib, seaborn
- scikit-learn, imbalanced-learn, XGBoost, SHAP
- openpyxl (to read the Excel input)

Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost shap openpyxl notebook
```

Exact dependency versions are not pinned in the notebook. For reproducibility, record tested versions in a `requirements.txt` or environment lockfile after verifying a working environment.

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repository>.git
   cd <your-repository>
   ```

   Replace the placeholders with your actual GitHub repository path.

2. Download the source dataset from [Zenodo](https://zenodo.org/records/7669215) and prepare a compatible `DATA_CLEAN.xlsx` file. The spreadsheet must include `Severity`, `Diagnosis`, and the input features used for training. The notebook maps the string values `complicated` and `uncomplicated` to binary labels; other values require preprocessing changes.

3. Place `DATA_CLEAN.xlsx` alongside `XGB_severity_model.ipynb`.

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `XGB_severity_model.ipynb` and run the cells in order. Training and SHAP computation may take time depending on your hardware and package versions.

## Modeling Workflow

### 1. Target and feature preparation

The notebook removes columns with **90% or more missingness**, maps severity labels to integers, and excludes the `Diagnosis` and `Severity` columns from predictors.

```python
df = pd.read_excel("DATA_CLEAN.xlsx")
df = df.loc[:, df.isnull().mean() < 0.9]
df["Severity"] = df["Severity"].map({
    "complicated": 1,
    "uncomplicated": 0
})
X = df.drop(columns=["Diagnosis", "Severity"])
y = df["Severity"]
```

### 2. Data splitting

A stratified **80% training / 20% test** split is used with `random_state=42`.

| Split | Total | Uncomplicated | Complicated |
|---|---:|---:|---:|
| Training | 623 | 528 | 95 |
| Test | 156 | 132 | 24 |

### 3. Preprocessing and imbalance handling

A `ColumnTransformer` applies median imputation to numeric columns and most-frequent imputation plus one-hot encoding to categorical columns. An `imblearn` pipeline then applies SMOTE during model fitting before XGBoost. This keeps oversampling inside training/CV fitting rather than directly oversampling the held-out test set.

**Methodological caution:** The notebook determines which columns to drop for missingness *before* the train/test split. For a stricter evaluation, learn feature-selection rules on training data only. Also review SMOTE's suitability for the encoded feature space and compare against approaches such as class weighting without oversampling.

### 4. Hyperparameter optimization

The notebook runs `RandomizedSearchCV` with **40 candidate parameter combinations** and **five-fold stratified cross-validation**, optimizing ROC-AUC (200 CV fits).

Best parameters in the saved run:

| Parameter | Selected value |
|---|---:|
| `n_estimators` | 200 |
| `learning_rate` | 0.05 |
| `max_depth` | 2 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.6 |
| `scale_pos_weight` | 4 |
| `min_child_weight` | 5 |
| `gamma` | 0 |

**Best tuning CV ROC-AUC:** `0.9010`.

The classifier uses `objective='binary:logistic'`, `eval_metric='logloss'`, and `random_state=42`.

## Results

The following are **saved notebook outputs**, not newly reproduced measurements.

### Train and held-out test performance

| Metric | Training | Test |
|---|---:|---:|
| ROC-AUC | 0.9720 | **0.9246** |
| Average precision (PR-AUC as reported) | 0.8493 | **0.7367** |
| Brier score (lower is better) | 0.0750 | 0.0919 |
| Accuracy | 0.90 | 0.85 |
| Sensitivity (complicated class) | 0.9579 | **0.7917** |
| Specificity | 0.8845 | **0.8636** |
| Positive predictive value | 0.5987 | **0.5135** |
| Negative predictive value | 0.9915 | **0.9580** |

The notebook also computes a trapezoidal precision–recall curve area separately. The table's PR-AUC values refer to `average_precision_score`; these two measures should not be assumed numerically identical.

### Test confusion matrix

| Actual / Predicted | Uncomplicated (0) | Complicated (1) |
|---|---:|---:|
| Uncomplicated (0) | 114 (TN) | 18 (FP) |
| Complicated (1) | 5 (FN) | 19 (TP) |

At the default classification threshold of **0.5**, the model identified 19 of 24 complicated cases and incorrectly classified 18 uncomplicated cases as complicated. Positive predictive value was approximately 51%, highlighting the cost of false-positive predictions in this imbalanced dataset.

### Additional cross-validation metrics

The notebook separately performs five-fold cross-validation on the selected best estimator using the training subset:

| Metric | Mean ± standard deviation |
|---|---|
| Sensitivity | 0.7579 ± 0.0537 |
| Specificity | 0.8655 ± 0.0380 |
| PPV | 0.5126 ± 0.0793 |
| NPV | 0.9518 ± 0.0116 |

These are **post-selection CV estimates using the tuned configuration**, not nested cross-validation or independent external validation.

## Model Explainability with SHAP

The notebook uses `shap.TreeExplainer` with the fitted XGBoost classifier and transformed test features. Included analyses are:

- SHAP summary plot for overall feature contribution patterns
- Mean absolute SHAP importance ranking (top 15 features displayed)
- Force plot for an individual prediction
- Dependence plot for the top-ranked feature
- Waterfall plot for an individual prediction
- Decision plot for the first 50 test observations

SHAP values describe model behavior and associations, **not causal clinical effects**. Interpretability does not establish clinical validity.

## Evaluation Visualizations

The notebook produces a confusion-matrix heatmap, ROC curve, and precision–recall curve, along with training/test classification reports. These are generated during execution; no exported figure files are assumed to exist in the repository.

## Reproducibility and Limitations

- **Input availability:** The notebook expects a local cleaned spreadsheet. A complete raw-to-clean data preparation script is not included.
- **Internal validation only:** The saved run uses one held-out split of 156 cases; it does not demonstrate external, multicenter, or prospective generalization.
- **Small positive test group:** Only 24 complicated cases are present in the test set, so reported sensitivity and PPV may be unstable.
- **Potential preprocessing leakage:** The missingness-based column filter is calculated before the split.
- **Threshold selection:** Classification metrics use a default 0.5 probability threshold; no clinical threshold-optimization study is provided.
- **Calibration:** A Brier score is reported, but a complete calibration analysis or independent calibration procedure is not documented.
- **Feature validity:** Model inputs and any cleaning or feature engineering must be reviewed for availability at the intended prediction time and for possible information leakage.
- **Tuning evaluation:** Hyperparameter selection and post-selection CV share the training data; nested CV would offer a less optimistic estimate of selection performance.
- **No deployment:** The notebook does not include a production inference API, model registry, clinical user interface, or monitoring system.
- **Clinical use:** Results are exploratory and must not be treated as a medical device or diagnostic recommendation.

## Possible Future Improvements

1. Add an explicit raw-to-clean data preparation script and versioned data dictionary.
2. Move all learned preprocessing and feature selection into fold-specific training pipelines.
3. Evaluate external and temporal validation cohorts and report uncertainty intervals.
4. Compare XGBoost against transparent baselines and other classifiers using identical splits.
5. Investigate class imbalance strategies and decision thresholds based on intended clinical use.
6. Add calibration plots, subgroup analyses, and robust model reporting.
7. Export reproducible model artifacts and pinned dependency versions after validation.

## Dataset Attribution

**Regensburg Pediatric Appendicitis Dataset.** Zenodo record 7669215.  
Dataset landing page: https://zenodo.org/records/7669215

For formal academic use, consult the Zenodo record for the original creators, dataset citation, version, and license, and cite the dataset accordingly. This project is an independent analysis of that dataset; it does not claim ownership of the source data.

## License

No license for this model repository has been specified. Add a `LICENSE` file if you intend to grant reuse permissions for your code, and independently respect the dataset's license and any privacy or research-use restrictions.

## Disclaimer

**For educational and research purposes only.** The results describe retrospective notebook evaluation and are not evidence that the model is safe, accurate, or appropriate for individual patient care. Clinical decisions must remain with qualified healthcare professionals using validated tools and clinical judgment.

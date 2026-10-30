# 🏡 Ames Housing Price Prediction — Advanced Regression Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Kaggle RMSLE](https://img.shields.io/badge/Best%20Public%20Score-0.11894-success.svg)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline developed for the Kaggle **"House Prices: Advanced Regression Techniques"** competition. The solution emphasizes **clean, leak-free statistical modeling**, surgical missing data imputation, domain-specific feature engineering, and a diverse 10-Fold Bagged Ensemble—achieving a competitive Public Leaderboard RMSLE of **`0.11894`** without relying on external data leakage.

---

## 📌 Executive Summary & Architecture

```
Raw Data (Train / Test)
   │
   ├──> Surgical Preprocessing (Outliers, Targeted Imputation, Test Set Winsorizing)
   │
   ├──> Feature Engineering (Neighborhood Tiers, Safe Quality Interactions, Total SF)
   │
   ├──> Statistical Transformations (Log1p Target, Positive-Only Box-Cox λ=0.15)
   │
   ├──> 10-Fold Cross-Validation & Out-Of-Fold (OOF) Prediction
   │       ├── SVR (Kernel RBF, C=20) ─────────────── [Weight: 30%]
   │       ├── XGBoost Regressor (Depth=3) ────────── [Weight: 25%]
   │       ├── ElasticNetCV (L1-Ratio Tuning) ─────── [Weight: 20%]
   │       ├── LassoCV (L1 Regularization) ────────── [Weight: 15%]
   │       └── CatBoost Regressor (Depth=4) ───────── [Weight: 10%]
   │
   └──> Convex Combination Ensemble & Jensen's Inequality Bias Correction
```

---

## 🔬 Key Engineering Highlights

### 1. Surgical Data Cleaning & Test Set Integrity
* **Outlier Removal:** Pruned extreme atypical sales (`GrLivArea > 4000 sq ft` with `SalePrice < $300,000` — houses `524` and `1299`) from training data as recommended by the dataset creator.
* **Targeted `MSZoning` Imputation:** Imputed missing values in test samples (`IDs 2905, 2217, 2251, 1916`) based on actual zoning maps (`RM` and `C (all)`) rather than blanket mode imputation.
* **Garage Feature Alignment (`ID 2127` & `2577`):** Repaired `ID 2127` (had an actual garage with missing `GarageYrBlt`) by imputing house `YearBuilt` to prevent linear penalty divergence. Repaired `ID 2577` by zeroing nonexistent garage metrics.
* **Test Winsorizing (`ID 2593`):** Capped extreme test area metrics (`GrLivArea` and `TotalBsmtSF`) at training maximums to prevent linear models from predicting non-convergent, unbounded multi-million dollar prices.

### 2. Feature Engineering
* **`Neighborhood_Tier`:** Binned 25 neighborhoods into 4 distinct economic tiers based on median price-per-square-foot from training data, compressing high-cardinality categorical noise into a singular monotonic ordinal signal.
* **`Log_Qual_TotalSF`:** Formulated a safe, non-linear interaction feature $\log(1 + \text{OverallQual} \times \text{TotalSF})$ that captures premium finishing on large properties without triggering runaway linear predictions.
* **Consolidated Physical Features:**
  * $\text{TotalSF} = \text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$
  * $\text{TotalBath} = \text{FullBath} + 0.5 \times \text{HalfBath} + \text{BsmtFullBath} + 0.5 \times \text{BsmtHalfBath}$
  * Age indicators: `HouseAge`, `RemodAge`, and `GarageAge`.
* **Pruning Sparse Dummies:** Applied `pd.get_dummies(..., drop_first=False)` to maintain symmetric regularization shrinkage toward the intercept, while filtering out rare binary columns appearing fewer than 3 times in training.

### 3. Mathematical & Skew Corrections
* **Target Transform:** Logarithmic transformation $y = \log(1 + \text{SalePrice})$ to minimize relative percentage error and directly optimize the competition RMSLE metric.
* **Selective Box-Cox Transformation:** Applied `boxcox1p(x, 0.15)` exclusively to continuous area/length features exhibiting positive right-tailed skew ($\text{skew} > 0.75$), preserving negative-skew ordinal scales.
* **Jensen's Inequality Compensation:** Compensated for median-to-mean prediction deflation arising from $\mathbb{E}[\exp(X)] \ge \exp(\mathbb{E}[X])$ via a post-processing smearing factor ($1.005 \times \text{expm1}(\hat{y})$).

---

## 🏆 Leaderboard Progression & Experiment Log

| Model / Submission File | Kaggle Public RMSLE | Validation OOF RMSLE | Verdict & Key Finding |
| :--- | :---: | :---: | :--- |
| **`submission.csv`** | `0.12100` | `~0.1110` | Initial baseline ensemble of linear models and trees. |
| **`submission_pure_stack.csv`** | `0.12254` | `0.1068` (Overfitted) | **Rejected:** Meta-model suffered from severe collinearity among log-predictions. |
| **`submission_kfold_blend.csv`** | `0.12153` | `0.1078` | Shifted to 10-Fold Bagged convex blending; eliminated meta-overfitting. |
| **`submission_master.csv`** | `0.11929` | `0.1055` | **Major Milestone:** Capped test area outliers & fixed Box-Cox skew logic. |
| **`submission_master_calibrated.csv`** | `0.11963` | — | **Rejected:** Manual tail stretching degraded native model calibration. |
| **`submission_robust_blend.csv`** | `0.11915` | `0.1062` | Replaced greedy Scipy weights with balanced 5-model distribution. |
| **`submission_v3_features.csv`** | **`0.11894`** | `0.1060` | **Best Current Submission:** Added `Neighborhood_Tier` and `Log_Qual_TotalSF`. |
| **`submission_v4_context.csv`** | `0.11946` | `0.1064` | **Rejected:** Granular neighborhood ratios introduced redundancy & noise. |
| **`submission_pseudo_labeled.csv`** | `0.11900` | — | **Rejected:** Pseudo-labeling top confident samples led to confirmation bias. |

---

## 🛠 Model Ensembling Strategy

The final predictor is a **10-Fold Bagged Convex Linear Blend** over 5 structurally diverse algorithms:

| Model Architecture | Hyperparameters / Configuration | Blend Weight |
| :--- | :--- | :---: |
| **Support Vector Regressor (SVR)** | `RobustScaler()`, `kernel='rbf'`, `C=20`, `epsilon=0.008`, `gamma=0.0003` | **30%** |
| **XGBoost Regressor** | `n_estimators=1200`, `learning_rate=0.02`, `max_depth=3`, `subsample=0.75` | **25%** |
| **ElasticNetCV** | `RobustScaler()`, `l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]`, `max_iter=15000` | **20%** |
| **LassoCV** | `RobustScaler()`, `alphas=logspace(-4.5, -1, 40)`, `max_iter=15000` | **15%** |
| **CatBoost Regressor** | `n_estimators=1200`, `learning_rate=0.02`, `depth=4`, `l2_leaf_reg=3` | **10%** |

---

## 🚀 Setup & Reproduction

### Prerequisites
* Python 3.10+
* Virtual environment (`venv` recommended)

### 1. Clone Repository
```bash
git clone [https://github.com/sukuinopain/house-prices-regression.git](https://github.com/sukuinopain/house-prices-regression.git)
cd house-prices-regression
```

### 2. Environment Setup
```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Competition Data
Download `train.csv` and `test.csv` from the [Kaggle Competition Page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place them in the root directory.

### 4. Run Pipeline
Open and execute the master notebook:
```bash
jupyter notebook "House_Prices_Solution.ipynb"
```

---

## 📂 Repository Structure

```
├── train.csv                      # Kaggle training dataset (1460 rows)
├── test.csv                       # Kaggle test dataset (1459 rows)
├── submission_v3_features.csv      # Best submission file (RMSLE: 0.11894)
├── submission_breakthrough.csv    # Final cross-blend with Jensen's correction
├── House_Prices_Solution.ipynb    # Complete pipeline code
├── .gitignore                     # Git exclusion rules (venv, cache)
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## 📜 Key Takeaways
1. **Targeted Imputation Beats Global Imputation:** Injecting domain-level fixes on test edge cases (`MSZoning`, `GarageYrBlt`) provides a tangible boost over naive median/mode filling.
2. **Restricting Collinear Overfitting:** Regularized stacking meta-models tend to overfit out-of-fold predictions when models have highly correlated outputs ($r > 0.98$). Robust weighted blending consistently generalizes better to the unseen test set.
3. **Bounded Interactions:** Multiplicative interaction features (`OverallQual * TotalSF`) must be bounded or logarithmically compressed to protect linear models from non-convergent predictions on test outliers.
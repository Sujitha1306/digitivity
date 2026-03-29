# Customer Churn Prediction - Bank Dataset
---
## What This Project Does

predict which bank customers are about to leave, before they actually do. If the bank knows who's at risk, they can step in early with offers or outreach instead of losing the customer entirely.
I used a dataset of 10,000 bank customers and trained three ML models on it. I also built a separate rule-based scoring engine.

---

## Dataset

**Source:** Bank Customer Churn Prediction Dataset  
**Records:** 10,000 customers  
**Features:** 12 columns (11 features + 1 target)

| Assignment Field | Dataset Column       | Type        | Description                 |
|------------------|----------------------|-------------|-----------------------------|
| Age              | `age`                | Numerical   | Customer age in years       |
| Income           | `estimated_salary`   | Numerical   | Estimated annual salary     |
| Purchases        | `products_number`    | Numerical   | Number of bank products used|
| Membership       | `membership`         | Categorical | Premium / Standard / Basic  |
| Churn            | `churn`              | Binary      | 1 = Churned, 0 = Retained   |

**Class Imbalance:** 79.6% not churned vs 20.4% churned, handled using SMOTE.

---

## Project Structure

```
churn-prediction/
│
├── churn_prediction.ipynb       ← Main Jupyter Notebook
├── Bank_Customer_Churn_Prediction.csv  ← Dataset
├── README.md                    ← This file
│
└── results/
    ├── 01_churn_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_smote_balance.png
    ├── 04_confusion_matrices.png
    ├── 05_roc_curves.png
    ├── 06_feature_importance.png
    ├── 07_risk_scoring_engine.png
    └── model_comparison.csv
```

---

## Methodology

### 1. Data Preprocessing

First pass was cleaning nulls, duplicates, empty strings masquerading as nulls.

The balance column was interesting though. A huge chunk of customers had zero balance, and when I dug in, they were churning at noticeably higher rates. So I added a zero_balance_flag binary column to make sure the models could pick up on that signal explicitly rather than hoping they'd figure it out.

I also checked skewness on the financial columns and log-transformed anything with absolute skewness above 0.75.

The membership column doesn't exist in the raw data so I engineered it from active_member and products_number:

Premium - active + multiple products
Standard - active + fewer products
Basic - inactive

### 2. Class Imbalance : SMOTE

80/20 split between retained and churned customers. This is the classic trap a model that just predicts "not churned" every time would hit 80% accuracy and look great on paper while being completely useless.
I used SMOTE to oversample the minority class, but only on the training set. The test set stays untouched.

### 3. Models Trained

**Logistic Regression** : Baseline model. Interpretable and fast. Useful as a reference point and in regulated environments where explainability is required.

**Random Forest** : Ensemble model using 100–200 decision trees. Hyperparameter tuning was applied using GridSearchCV with 5-fold cross-validation, optimizing for F1 score. Handles non-linear feature interactions naturally.

**XGBoost** : Gradient boosting model. Tuned with GridSearchCV across learning rate, depth, and subsampling parameters. Typically the strongest performer on tabular classification tasks.

### 4. Evaluation Metrics
Models were evaluated on: Accuracy, Precision, Recall, F1-Score, and ROC-AUC. For churn prediction, Recall is prioritized because missing a churner (false negative) is more costly to the business than a false alarm (false positive). ROC-AUC measures the model's overall discriminatory power across all thresholds.

### 5. Rule-Based Churn Risk Scoring Engine
Beyond ML models, a custom scoring engine was built that assigns each customer a Churn Risk Score from 0 to 100 based on weighted business rules. Factors considered include age, account balance, product usage, active membership status, credit score, tenure, and geography.

Customers are classified into four tiers - Low (0–25), Medium (26–50), High (51–75), and Critical (76–100). The engine was validated by comparing actual churn rates per tier churn rate consistently increased with risk score, confirming the engine is meaningful and not arbitrary.

This approach reflects real-world banking practice where ML predictions need to be explainable to non-technical stakeholders and regulators.

---

## Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| XGBoost             | 0.8468   | 0.6401    | 0.5564 | 0.5953   | 0.8360  |
| Random Forest       | 0.8339   | 0.5875    | 0.6026 | 0.5949   | 0.8393  |
| Logistic Regression | 0.7445   | 0.4196    | 0.6821 | 0.5195   | 0.7874  |

**Best Model: Random Forest** marginally highest ROC-AUC (0.8393) and 
higher Recall (0.6026 vs 0.5564), making it the preferred production model 
where catching churners matters most. XGBoost is nearly identical in F1, 
both significantly outperforming the Logistic Regression baseline.

---

## How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place `Bank_Customer_Churn_Prediction.csv` in the project root
4. Open `churn_prediction.ipynb` in Jupyter Notebook or JupyterLab
5. Run all cells in order (Kernel → Restart & Run All)
6. Results will be saved automatically to the `results/` folder

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
jupyter
```

---

## Key Findings

1. Age and account balance are the strongest predictors of churn- older customers and those with zero balance are most likely to leave.
2. Inactive members (active_member = 0) show significantly higher churn rates regardless of other features.
3. Customers using 3 or more bank products paradoxically have higher churn likely because over-sold customers feel overwhelmed.
4. Germany-based customers churn at a higher rate than France or Spain customers in this dataset.
5. SMOTE was essential without balancing, models would predict "not churned" for nearly all customers and still achieve 80% accuracy, which is misleading.

---


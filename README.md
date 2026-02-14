# 🏦 Credit Card Default Prediction

![alt text](Credit-card-Default.png)

## 📊 Project Overview

This project focuses on predicting credit card default payments using various machine learning algorithms. The dataset contains information about credit card clients in Taiwan from April 2005 to September 2005, and we aim to predict which clients will default on their payments in the following month.

## 🤖 Binary Classification Models

This project covers Binary Classification using the following models:

1. **Logistic Regression** - Linear model for binary classification
2. **Decision Tree Classifier** - Tree-based model with interpretable decision rules
3. **K-Nearest Neighbor Classifier** - Instance-based learning algorithm
4. **Naive Bayes Classifier** - Gaussian or Multinomial probabilistic classifier
5. **Ensemble Model - Random Forest** - Bagging ensemble of decision trees
6. **Ensemble Model - XGBoost** - Gradient boosting ensemble method



## �📁 Dataset Description

**Dataset**: Default of Credit Card Clients Dataset

**Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

### Dataset Information
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

# Note: Daaset intial source to Kaggle was from UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

### Content
There are 25 variables: 

| Variable | Description |
|----------|-------------|
| **ID** | ID of each client |
| **LIMIT_BAL** | Amount of given credit in NT dollars (includes individual and family/supplementary credit) |
| **SEX** | Gender (1=male, 2=female) |
| **EDUCATION** | (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown) |
| **MARRIAGE** | Marital status (1=married, 2=single, 3=others) |
| **AGE** | Age in years |
| **PAY_0** | Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above) |
| **PAY_2** | Repayment status in August, 2005 (scale same as above) |
| **PAY_3** | Repayment status in July, 2005 (scale same as above) |
| **PAY_4** | Repayment status in June, 2005 (scale same as above) |
| **PAY_5** | Repayment status in May, 2005 (scale same as above) |
| **PAY_6** | Repayment status in April, 2005 (scale same as above) |
| **BILL_AMT1** | Amount of bill statement in September, 2005 (NT dollar) |
| **BILL_AMT2** | Amount of bill statement in August, 2005 (NT dollar) |
| **BILL_AMT3** | Amount of bill statement in July, 2005 (NT dollar) |
| **BILL_AMT4** | Amount of bill statement in June, 2005 (NT dollar) |
| **BILL_AMT5** | Amount of bill statement in May, 2005 (NT dollar) |
| **BILL_AMT6** | Amount of bill statement in April, 2005 (NT dollar) |
| **PAY_AMT1** | Amount of previous payment in September, 2005 (NT dollar) |
| **PAY_AMT2** | Amount of previous payment in August, 2005 (NT dollar) |
| **PAY_AMT3** | Amount of previous payment in July, 2005 (NT dollar) |
| **PAY_AMT4** | Amount of previous payment in June, 2005 (NT dollar) |
| **PAY_AMT5** | Amount of previous payment in May, 2005 (NT dollar) |
| **PAY_AMT6** | Amount of previous payment in April, 2005 (NT dollar) |
| **default_payment_next_month** | Default payment (1=yes, 0=no) |

## 🔧 Feature Engineering

The following engineered features were created to improve model performance:

| Feature | Description |
|---------|-------------|
| **TOTAL_BILL_AMT** | Total bill amount across all 6 months |
| **TOTAL_PAY_AMT** | Total payment amount across all 6 months |
| **AVG_BILL_AMT** | Average bill amount |
| **AVG_PAY_AMT** | Average payment amount |
| **PAY_TO_BILL_RATIO** | Ratio of total payments to total bills |
| **WORST_PAYMENT_STATUS** | Worst (highest) payment delay status |
| **CREDIT_UTILIZATION** | Credit utilization ratio (latest bill / credit limit) |
| **AGE_GROUP** | Age categorized into groups (Young, Adult, Middle, Senior, Elder) |

## 📊 Evaluation Metrics

Models are evaluated on the **training set** with the following metrics:

- **Accuracy** - Overall prediction accuracy
- **AUC Score** - Area under ROC curve
- **Precision** - Positive predictive value
- **Recall** - Sensitivity/True positive rate
- **F1 Score** - Harmonic mean of precision and recall
- **MCC Score** - Matthews Correlation Coefficient (balanced measure)

Confusion matrices are also generated for visual evaluation of model performance.

## 💾 Exported Models & Artifacts

All trained models and preprocessing artifacts are exported to the `model/` folder:

### Trained Models (`.pkl` files)
- `Logistic_Regression.pkl`
- `Decision_Tree_Classifier.pkl`
- `K_Nearest_Neighbor_Classifier.pkl`
- `Naive_Bayes_Classifier_Gaussian.pkl`
- `Ensemble_Model_Random_Forest.pkl`
- `Ensemble_Model_XGBoost.pkl`

### Preprocessing Artifacts
- `scaler.pkl` - StandardScaler fitted on training data
- `feature_names.json` - List of all feature column names
- `numerical_features.json` - List of numerical feature names

### Test Data (exported to `Dataset/` folder)
- `Credit_card_testdata.csv` - Original test features (before feature engineering, includes ID column)
- `Credit_card_testlabels.csv` - Test labels (target variable)

## 📈 Model Performance Observations

Based on the training set evaluation, here are the observations for each model:

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Moderate accuracy (80.7%) with good AUC (0.74). Shows high precision (69%) but low recall (23%), indicating it's conservative in predicting defaults - when it predicts default, it's usually correct, but misses many actual defaults. Best for minimizing false positives. |
| **Decision Tree** | Very poor performance with low accuracy (22.5%) and negative MCC (-0.27). High recall (66%) but extremely low precision (17%), meaning it over-predicts defaults. Likely overfitting to training data. |
| **kNN** | Best overall performance with highest accuracy (84.4%) and AUC (0.88). Good balance of precision (72%) and recall (48%) with strong F1 score (0.57). Most reliable model for this dataset. |
| **Naive Bayes** | Low accuracy (42.3%) but highest recall (88%) for detecting defaults. Very low precision (26%) indicates many false positives. Good for catching potential defaulters but with high false alarm rate. |
| **Random Forest (Ensemble)** | Disappointing performance with moderate accuracy (67.1%) and low AUC (0.57). Low precision (25%) and recall (24%) suggest the model is struggling with this dataset. May need hyperparameter tuning. |
| **XGBoost (Ensemble)** | Moderate accuracy (67.5%) with AUC of 0.61. Better recall (43%) than Random Forest but still low precision (32%). Performs better than Random Forest but below expectations for a gradient boosting model. |

### Key Insights:
- **kNN is the top performer** with the best balance of all metrics
- **Logistic Regression** is the most conservative model with highest precision
- **Naive Bayes** catches the most defaults (highest recall) but with many false alarms
- **Decision Tree and Random Forest** show signs of poor generalization
- **Ensemble methods** (Random Forest, XGBoost) underperformed compared to kNN, suggesting they may need hyperparameter optimization

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/sureshbabuveluswamy/Credit-Card-Default.git
cd Credit-Card-Default
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run model/creditcard_streamlit.py
```

### Dependencies

- **streamlit** : Web application framework
- **scikit-learn** : Machine learning library
- **numpy** : Numerical computing
- **pandas** : Data manipulation and analysis
- **matplotlib** : Data visualization
- **seaborn** : Statistical data visualization
- **xgboost** : Gradient boosting library
- **joblib** : Model serialization

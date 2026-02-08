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

## 📁 Dataset Description

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
| **default.payment.next.month** | Default payment (1=yes, 0=no) |


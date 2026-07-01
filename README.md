## Credit Card Fraud Detection ##

## Problem Statement ##

Credit card fraud is a major challenge for financial institutions, leading to significant financial losses and security risks. Since fraudulent transactions are rare compared to legitimate ones, accurately identifying them is difficult.

The goal of this project is to develop a machine learning model that can classify credit card transactions as fraudulent or legitimate. The project also demonstrates an end-to-end MLOps workflow, including data versioning, experiment tracking, model deployment, and cloud deployment using modern tools.


# 💳 Credit Card Fraud Detection

## 📌 Problem Statement

Credit card fraud is one of the biggest challenges faced by financial institutions, resulting in significant financial losses and security risks. Since fraudulent transactions represent only a small fraction of all transactions, building an accurate fraud detection system is a challenging imbalanced classification problem.

The primary objective of this project is to develop a machine learning model capable of accurately classifying credit card transactions as **fraudulent** or **legitimate**. The project focuses on developing strong machine learning skills through comprehensive exploratory data analysis, feature engineering, model development, and evaluation while incorporating domain knowledge using a hybrid fraud risk scoring approach.

---

# 📊 Exploratory Data Analysis (EDA)

A comprehensive exploratory data analysis was performed to understand the underlying characteristics of the dataset and identify important fraud patterns.

The analysis includes:

- Missing value analysis
- Duplicate value detection
- Target variable distribution
- Class imbalance analysis
- Numerical feature analysis
- Categorical feature analysis
- Correlation analysis
- Outlier detection
- Transaction amount analysis
- Device trust score analysis
- Foreign transaction analysis
- Velocity (Last 24 Hours) analysis
- Univariate, bivariate, and multivariate analysis

The insights obtained during EDA were used to guide feature engineering and model development.

---

# ⚙️ Feature Engineering

Feature engineering was performed to improve the predictive capability of the machine learning models by incorporating both statistical analysis and domain knowledge.

The feature engineering pipeline includes:

- Feature selection
- Handling class imbalance using SMOTE
- Multicollinearity analysis using VIF
- Outlier analysis
- Business rule-based feature creation
- Hybrid fraud risk scoring approach
- Data preprocessing for model training

A **hybrid fraud detection approach** was designed by combining:

- Rule-based business logic
- Machine learning fraud probability

to generate an overall **Fraud Risk Score**, making predictions more interpretable and closer to real-world fraud detection systems.

---

# 🤖 Machine Learning

The following machine learning models were implemented and compared:

- Logistic Regression
- Random Forest
- XGBoost

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- Confusion Matrix

The best-performing model was selected based on its ability to detect fraudulent transactions while maintaining a balance between precision and recall.

---

# 📌 Conclusion

- Comprehensive EDA helped identify important fraud patterns and guided feature engineering.
- Feature engineering significantly improved model performance.
- Both SMOTE and non-SMOTE models produced competitive results.
- The hybrid approach combining business rules with machine learning improved model interpretability.
- This project demonstrates practical machine learning techniques for solving an imbalanced classification problem using real-world inspired fraud detection strategies.

---


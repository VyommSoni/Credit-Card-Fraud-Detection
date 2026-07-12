💳 Credit Card Fraud Detection using Machine Learning & MLOps
📌 Problem Statement

Credit card fraud is one of the biggest challenges faced by financial institutions, resulting in significant financial losses and security risks. Since fraudulent transactions represent only a small fraction of all transactions, building an accurate fraud detection system is a challenging imbalanced classification problem.

The objective of this project is to develop a production-ready machine learning system capable of accurately classifying credit card transactions as fraudulent or legitimate. The project demonstrates an end-to-end MLOps workflow, including data validation, feature engineering, experiment tracking, model evaluation, deployment, and cloud model storage.

📊 Exploratory Data Analysis (EDA)

Comprehensive exploratory data analysis was performed to understand fraud patterns and identify important insights before model development.

The analysis includes:

Missing value analysis
Duplicate value detection
Target distribution analysis
Class imbalance analysis
Numerical feature analysis
Categorical feature analysis
Correlation analysis
Outlier detection
Transaction amount analysis
Device trust score analysis
Foreign transaction analysis
Transaction velocity (Last 24 Hours)
Univariate analysis
Bivariate analysis
Multivariate analysis

The insights obtained during EDA guided the feature engineering and model development process.

⚙️ Feature Engineering

Feature engineering was performed using both statistical analysis and domain knowledge to improve predictive performance.

The feature engineering pipeline includes:

Feature selection
Missing value handling
Class imbalance handling using SMOTE
Multicollinearity analysis using VIF
Outlier analysis
Business rule engineering
Data preprocessing
Hybrid Fraud Risk Score generation

A hybrid fraud detection strategy combines:

Rule-based business logic
Machine learning fraud probability

to generate an overall Fraud Risk Score, making predictions more interpretable and closer to real-world fraud detection systems.

🤖 Machine Learning

The following machine learning models were developed and compared:

Logistic Regression
Random Forest
XGBoost

The models were evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix

The best-performing model was selected based on its fraud detection capability while maintaining a balance between precision and recall.

🚀 Hybrid Fraud Detection Architecture

Instead of relying on a single machine learning model, this project implements a hybrid fraud detection architecture inspired by real-world banking systems.

The workflow is:

Business Rule Engine generates rule-based fraud risk.
Logistic Regression predicts fraud probability.
Rule score and probability are combined to create a Fraud Risk Score.
Random Forest uses the engineered Fraud Risk Score along with transaction features to make the final prediction.

This layered architecture improves both model interpretability and prediction performance.

⚙️ End-to-End MLOps Pipeline

This project follows a modular production-style MLOps architecture.

The pipeline consists of:

Data Ingestion
Data Validation
Data Transformation
Model Training
Model Evaluation
Model Pusher
Prediction Pipeline

Each stage is implemented as an independent component using configuration-driven architecture and artifact-based communication.

🛠 MLOps Tools & Technologies

The project integrates modern MLOps tools used in production environments.

Machine Learning
Scikit-Learn
XGBoost
Pandas
NumPy
MLOps
MLflow (Experiment Tracking)
DVC (Data & Model Versioning)
Git & GitHub
Docker
FastAPI
AWS S3
Logging
Custom Exception Handling
☁️ Deployment

The production model is automatically pushed to an AWS S3 bucket.

The prediction workflow is:

Load the latest production model from AWS S3.
Apply preprocessing using the saved transformer.
Predict fraud probability using the Logistic Regression base model.
Generate the hybrid Fraud Risk Score.
Predict the final fraud class using the Random Forest model.
Return predictions through a FastAPI REST API.

📂 Project Architecture
Data Ingestion
        │
        ▼
Data Validation
        │
        ▼
Data Transformation
        │
        ▼
Business Rules
        │
        ▼
Logistic Regression
(Base Probability Model)
        │
        ▼
Fraud Risk Score
        │
        ▼
Random Forest
(Final Classifier)
        │
        ▼
Model Evaluation
        │
        ▼
Model Pusher (AWS S3)
        │
        ▼
FastAPI Prediction Service


💡 Key Highlights
1.) End-to-end production-ready MLOps pipeline.
2.) Hybrid fraud detection using business rules and machine learning

Two-model architecture:

- Logistic Regression for fraud probability estimatior
- Random Forest for final fraud classification.
- Modular and reusable pipeline architecture.
- Experiment tracking with MLflow.
- Dataset and model versioning with DVC.
- AWS S3 integration for production model storage.
- FastAPI REST API for real-time inference.
- Docker-ready deployment.
- Configuration-driven project structure.
- Logging and custom exception handling.
- Cloud deployment using modern MLOps practices.

 📌 Conclusion

This project demonstrates how machine learning can be combined with business rules to build a realistic fraud detection system. Beyond model development, it showcases an end-to-end MLOps workflow, including experiment tracking, version control, cloud deployment, and scalable prediction pipelines, providing experience with production-oriented machine learning practices.
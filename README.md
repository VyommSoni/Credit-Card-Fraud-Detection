# 💳 Credit Card Fraud Detection

An end-to-end production-ready Machine Learning and MLOps project for detecting fraudulent credit card transactions using a hybrid fraud detection architecture, modern MLOps tools, and cloud deployment.

---

# 📌 Problem Statement

Credit card fraud is one of the biggest challenges faced by financial institutions, resulting in significant financial losses and security risks.

Since fraudulent transactions represent only a small fraction of all transactions, building an accurate fraud detection system becomes a challenging **imbalanced classification** problem.

The objective of this project is to develop a production-ready machine learning system capable of accurately classifying credit card transactions as **Fraudulent** or **Legitimate** while following industry-standard MLOps practices.

The project demonstrates the complete lifecycle of an ML system, from data preprocessing and feature engineering to model deployment on AWS.

---

# 📊 Exploratory Data Analysis (EDA)

Comprehensive exploratory data analysis was performed to understand fraud patterns and identify meaningful insights before model development.

### The analysis includes:

* Missing Value Analysis
* Duplicate Value Detection
* Target Distribution Analysis
* Class Imbalance Analysis
* Numerical Feature Analysis
* Categorical Feature Analysis
* Correlation Analysis
* Outlier Detection
* Transaction Amount Analysis
* Device Trust Score Analysis
* Foreign Transaction Analysis
* Transaction Velocity (Last 24 Hours)
* Univariate Analysis
* Bivariate Analysis
* Multivariate Analysis

The insights obtained during EDA guided feature engineering and model development.

---

# ⚙️ Feature Engineering

Feature engineering combines statistical techniques with business domain knowledge to improve predictive performance.

### The feature engineering pipeline includes:

* Feature Selection
* Missing Value Handling
* Outlier Detection
* Multicollinearity Analysis (VIF)
* Handling Class Imbalance using SMOTE
* Business Rule Engineering
* Hybrid Fraud Risk Score Generation
* Data Preprocessing using Scikit-Learn Pipelines

A hybrid fraud detection strategy combines:

* Rule-Based Business Logic
* Machine Learning Fraud Probability

to generate an overall **Fraud Risk Score**, making predictions more interpretable and closer to real-world banking fraud detection systems.

---

# 🤖 Machine Learning

The following Machine Learning algorithms were implemented and compared:

* Logistic Regression
* Random Forest
* XGBoost

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The best-performing model was selected based on fraud detection capability while maintaining a balance between precision and recall.

---

# 🚀 Hybrid Fraud Detection Architecture

Unlike traditional fraud detection systems that rely on a single machine learning model, this project implements a **Hybrid Fraud Detection Architecture** inspired by real-world financial institutions.

### Workflow

1. Apply Business Rules on each transaction.
2. Predict Fraud Probability using Logistic Regression.
3. Generate a Hybrid Fraud Risk Score by combining:

   * Business Rule Score
   * Machine Learning Probability
4. Train a Random Forest model using the engineered Fraud Risk Score along with transaction features.
5. Generate the final fraud prediction.

This layered architecture improves both prediction accuracy and model interpretability.

---

# ⚙️ End-to-End MLOps Pipeline

This project follows a modular production-ready MLOps architecture.

The complete pipeline includes:

* Data Ingestion
* Data Validation
* Data Transformation
* Model Training
* Model Evaluation
* Model Pusher
* Prediction Pipeline

Each stage is implemented as an independent component using configuration files and artifact-based communication.

---

# 🛠️ MLOps Tools & Technologies

## Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost

## MLOps

* MLflow (Experiment Tracking)
* DVC (Data & Model Versioning)
* Git & GitHub
* Docker
* FastAPI
* AWS S3
* Logging
* Custom Exception Handling
* Configuration-Based Pipeline Architecture

---

# ☁️ Deployment

The production-ready model is automatically pushed to an AWS S3 bucket.

### Prediction Pipeline

1. Load the latest production model from AWS S3.
2. Load the saved preprocessing pipeline.
3. Preprocess incoming transaction data.
4. Predict fraud probability using the Logistic Regression Base Model.
5. Generate the Hybrid Fraud Risk Score.
6. Perform final prediction using the Random Forest model.
7. Return fraud prediction through a FastAPI REST API.

---

# 📂 Project Architecture

```text
                 Raw Dataset
                      │
                      ▼
             Data Ingestion
                      │
                      ▼
             Data Validation
                      │
                      ▼
          Data Transformation
                      │
                      ▼
           Business Rule Engine
                      │
                      ▼
     Logistic Regression Model
      (Fraud Probability)
                      │
                      ▼
        Hybrid Fraud Risk Score
                      │
                      ▼
      Random Forest Classifier
        (Final Prediction)
                      │
                      ▼
            Model Evaluation
                      │
                      ▼
              Model Pusher
                (AWS S3)
                      │
                      ▼
         FastAPI Prediction API
```

---

# 📁 Project Structure

```text
src/
│
├── Components
│   ├── DataIngestion
│   ├── DataValidation
│   ├── DataTransformation
│   ├── ModelTrainer
│   ├── ModelEvaluation
│   └── ModelPusher
│
├── Business_Rules
│
├── Pipeline
│
├── Infrastructure
│
├── Configuration
│
├── Utility
│
├── Exception
│
├── Logging
│
└── Components/ML
    ├── estimator.py
    └── S3_estimator.py
```

---

# 💡 Key Highlights

* End-to-End Production-Ready MLOps Pipeline
* Hybrid Fraud Detection using Business Rules & Machine Learning
* Two-Model Architecture

  * Logistic Regression for Fraud Probability Estimation
  * Random Forest for Final Fraud Classification
* Modular Pipeline Architecture
* Experiment Tracking using MLflow
* Dataset & Model Versioning using DVC
* AWS S3 Model Storage
* FastAPI REST API for Real-Time Prediction
* Docker Ready Deployment
* Configuration-Driven Project Structure
* Logging & Custom Exception Handling
* Cloud-Based Model Deployment

---

# 📌 Conclusion

This project demonstrates how machine learning and business rules can be combined to build a realistic fraud detection system.

Beyond model development, it showcases an end-to-end MLOps workflow including experiment tracking, data versioning, cloud model deployment, modular pipeline design, and production-ready prediction services.

The project reflects practical machine learning engineering skills by combining model development with scalable software engineering and MLOps best practices.

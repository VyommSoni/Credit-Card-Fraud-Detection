import os

Collection_name="Credit_card_Collection"
Data_FilePath="Data\credit_card_fraud_10k.csv"
Database_name="Fraud_recognization"


ARTIFACT_DIR='Artifacts'
PIPELINE_NAME="Fraud_Pipeline"
TARGET_COLUMN='is_fraud'

DATA_INGESTION_DIR="DataIngestion"
INGESTED_DATA_DIR="Ingested_data"
Feature_Store_Path="FeatureStore.csv"
TRAIN_FILE_PATH="Train.csv"
TEST_FILE_PATH="Test.csv"
TEST_SIZE=0.2

CONIFG_PATH=os.path.join("Config","Schema.yml")

DATA_VALIDATION_DIR="DataValidation"
VALIDATED_DIR="Valid_dir"
INVALIDATED_DIR="Invalid_dir"
VALID_TRAIN_FILEPATH="Valid_train.csv"
VALID_TEST_FILEPATH="Valid_test.csv"
INVALID_TRAIN_FILEPATH="Invalid_train.csv"
INVALID_TEST_FILEPATH="Invalid_test.csv"



DATA_TRANSFORMATION_DIR='DataTransformation'
PREPROCESSOR_PATH="Preprocessor.pkl"
TRANSFORMED_DIR='Transformed_dir'
TRANSFORMED_TRAIN_FILE_PATH='Train.nparray'
TRANSFORMED_TEST_FILE_PATH='Test.nparray'

COLUMNS_TO_DROP=['transaction_id','merchant_category','cardholder_age']

MODEL_DIR='MODEL'
TRAINED_MODEL_DIR='Trained_model'
MODEL_PATH='Model.pkl'
EXPECTED_ACCURACY=0.70

BUCKET_NAME="frauddetection471165810235"


DAGSHUB_REPO_OWNER="svyom21"
DAGSHUB_REPO_NAME="Credit-Card-Fraud-Detection"
MLFLOW_TRACKING_URI="https://dagshub.com/svyom21/Credit-Card-Fraud-Detection.mlflow"

MODEL_WEIGHTAGE=0.6
BUISSNESS_RULES_WEIGHTAGE=0.4
SPLIT=5
best_model_path=os.path.join("Saved_Models","Model.pkl")




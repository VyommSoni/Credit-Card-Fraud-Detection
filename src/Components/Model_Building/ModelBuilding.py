import os
import sys
import numpy as np
import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from src.entity.Artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.Config_entity import ModelTrainerConfig
from src.Utility.Utilis.Utility import Utils
from src.Components.ML.estimator import FraudModel
from src.Components.ML.Metric import Metric
from src.Buissness_rules.Rules import Rules
from src.Constant.TrainingPipeline_Constant import *

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.utils = Utils()
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.rules_engine = Rules()
        # Explicitly defining our two hybrid stacking layers
        self.base_model = LogisticRegression(max_iter=1000, random_state=42)
        self.meta_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

        # Initialize Dagshub & MLflow configurations
        try:
            logging.info("Initializing MLflow and Dagshub tracking backend...")
            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER, 
                repo_name=DAGSHUB_REPO_NAME
            )
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("Hybrid_Stacking_Fraud_Engine")
        except Exception as e:
            logging.warning(f"MLflow initialization skipped or failed: {str(e)}. Proceeding with raw training.")

    def generate_oof_probabilities(self, X_train: np.array, y_train: np.array, X_test: np.array) -> tuple:
        """Generates clean out-of-fold probability features to completely prevent data leakage"""
        try:
            skf = StratifiedKFold(n_splits=SPLIT, shuffle=True, random_state=42)
            oof_train_probas = np.zeros(X_train.shape[0])
            oof_test_probas_meta = np.zeros((X_test.shape[0], skf.n_splits))

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                # Fit base model on training folds only to isolate holdout verification sets
                self.base_model.fit(X_train[train_idx], y_train[train_idx])
                
                # Predict on the holdout fold
                oof_train_probas[val_idx] = self.base_model.predict_proba(X_train[val_idx])[:, 1]
                
                # Capture test sets predictions across the split folds
                oof_test_probas_meta[:, fold] = self.base_model.predict_proba(X_test)[:, 1]

            return oof_train_probas.reshape(-1, 1), oof_test_probas_meta.mean(axis=1).reshape(-1, 1)
        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

    def Track_mlflow(self, train_metric, test_metric, best_model_path: str):
        """Logs all system evaluation parameters and models straight to MLflow dashboard"""
        try:
            logging.info("Logging model training run characteristics and artifacts to MLflow server...")
            
            # Start an active evaluation tracking session
            with mlflow.start_run(run_name="Hybrid_Stack_Trainer_Run"):
                # Track model types as parameters
                mlflow.log_param("base_layer_model", "LogisticRegression")
                mlflow.log_param("meta_layer_model", "RandomForestClassifier")
                mlflow.log_param("hybrid_blend_ratio", "60_AI_40_Rules")

                # Track metrics for the Training Dataset
                mlflow.log_metric("train_accuracy", train_metric.accuracy_score)
                mlflow.log_metric("train_f1_score", train_metric.f1score)
                mlflow.log_metric("train_precision", train_metric.precision_score)
                mlflow.log_metric("train_recall", train_metric.recall_score)

                # Track metrics for the Validation Test Dataset
                mlflow.log_metric("test_accuracy", test_metric.accuracy_score)
                mlflow.log_metric("test_f1_score", test_metric.f1score)
                mlflow.log_metric("test_precision", test_metric.precision_score)
                mlflow.log_metric("test_recall", test_metric.recall_score)

                # Log the final saved FraudModel container as an MLflow artifact
                mlflow.log_artifact(best_model_path, artifact_path="Packaged_Ensemble_Model")
                logging.info("MLflow logging completed successfully!")

        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

    def Initiate_ModelTraining(self, X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating Hybrid Stacking Model Training...")

            # 1. Load Transformed NumPy Arrays from your Transformation phase pathing
            train_arr = self.utils.load_numpy_array(filepath=self.data_transformation_artifact.Transformed_Trainarr)
            test_arr = self.utils.load_numpy_array(filepath=self.data_transformation_artifact.Transformed_Testarr)

            if train_arr is not None and test_arr is not None:
                X_train, Y_train = train_arr[:, :-1], train_arr[:, -1]
                X_test, Y_test = test_arr[:, :-1], test_arr[:, -1]

                # 2. Extract leak-free stacking probabilities via Base Model
                train_probas, test_probas = self.generate_oof_probabilities(X_train, Y_train, X_test)

                # 3. Process Business Rules using RAW dataframes
                train_rule_scores = X_train_raw.apply(self.rules_engine.rules, axis=1).values.reshape(-1, 1)
                test_rule_scores = X_test_raw.apply(self.rules_engine.rules, axis=1).values.reshape(-1, 1)

                # 4. Construct Blended Hybrid Risk Feature Columns
                train_hybrid_feature = (MODEL_WEIGHTAGE* (train_probas * 100)) + (BUISSNESS_RULES_WEIGHTAGE * train_rule_scores)
                test_hybrid_feature = (MODEL_WEIGHTAGE * (test_probas * 100)) + (BUISSNESS_RULES_WEIGHTAGE * test_rule_scores)

                # 5. Stack hybrid feature column directly onto your processed matrices
                X_train_stacked = np.hstack((X_train, train_hybrid_feature))
                X_test_stacked = np.hstack((X_test, test_hybrid_feature))

                # 6. Fit Meta-Model (Random Forest) on the complete stacked layout
                logging.info("Training Layer 2 Meta-Model...")
                self.meta_model.fit(X_train_stacked, Y_train)

                # 7. Final Fit of Base Model on 100% of standard features for production inference
                self.base_model.fit(X_train, Y_train)

                # 8. Generation of performance metrics
                y_train_pred = self.meta_model.predict(X_train_stacked)
                y_test_pred = self.meta_model.predict(X_test_stacked)
                best_model_score = accuracy_score(Y_test, y_test_pred)

                if best_model_score > EXPECTED_ACCURACY:
                    logging.info(f"Target score criteria satisfied: {best_model_score}")
                    
                    # Generate metrics using my project's custom Metric component wrapper
                    metric=Metric()
                    Train_metric = metric.calculate_metric(y_true=Y_train, y_pred=y_train_pred)
                    Test_metric = metric.calculate_metric(y_true=Y_test, y_pred=y_test_pred)

                    # 9. Pack everything together inside your custom FraudModel container
                    preprocessor = self.utils.load_object(filepath=self.data_transformation_artifact.Preprocessor_Path)
                    packaged_ensemble = FraudModel(preprocessor=preprocessor, base_model=self.base_model, meta_model=self.meta_model)

                    # Save the model object locally
                    os.makedirs(os.path.dirname(self.model_trainer_config.model_path), exist_ok=True)
                    self.utils.save_object(filepath=self.model_trainer_config.model_path, obj=packaged_ensemble)

                    # 10. Trigger the internal MLflow session to log everything
                    self.Track_mlflow(
                        train_metric=Train_metric, 
                        test_metric=Test_metric, 
                        best_model_path=self.model_trainer_config.model_path
                    )

                    model_trainer_artifact = ModelTrainerArtifact(
                        BestModelPath=self.model_trainer_config.model_path,
                        train_metric_artifact=Train_metric,
                        test_metric_artifact=Test_metric
                    )

                    return model_trainer_artifact
                else:
                    logging.info(f"Model accuracy {best_model_score} below threshold {EXPECTED_ACCURACY}")
            else:
                logging.warning("Training cancelled: Processed input arrays returned empty.")
            
        except Exception as e:
            raise CreditCradFraudDetection(e, sys)
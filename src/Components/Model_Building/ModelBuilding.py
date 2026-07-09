from src.Logging.fraud_logging import logging
from src.Exception.fraud_exception import CreditCradFraudDetection
import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from pandas import DataFrame
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.Constant.TrainingPipeline_Constant import *
from src.entity.Artifact_entity import DataTransformationArtifact,ModelTrainerArtifact ,ClassificationMetricArtifact
from src.entity.Config_entity import ModelTrainerConfig
from src.Utility.Utilis.Utility import Utils
from src.Components.ML.estimator import FraudModel
from src.Components.ML.Metric import Metric
import dagshub
import numpy as np


class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):
        self.utils=Utils()
        self.data_transformation_artifact=data_transformation_artifact
        self.model_trainer_config=model_trainer_config

        self.models={
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(),
            "Logistic Regression": LogisticRegression()

        }
    def evaluate_model(self,X_train,y_train,X_test,y_test,models):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]

                model.fit(X_train, y_train)  # Train model

                y_test_pred = model.predict(X_test)
                y_train_pred=model.predict(X_train)

                test_model_score = accuracy_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report
        
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

    def get_best_model(self,
                       x_train: np.array,
                       y_train: np.array,
                       x_test: np.array,
                       y_test: np.array):
        try:

            model_report= self.evaluate_models(
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=self.models
            )


            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object = self.models[best_model_name]

            best_model_object.fit(x_train,y_train)

            y_test_pred=best_model_object.predict(x_test)
            y_train_pred=best_model_object.predict(x_train)


            return best_model_name, best_model_object,best_model_score,y_test_pred,y_train_pred
        
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
    def Track_mlflow(self,best_model,classification_metric:ClassificationMetricArtifact):
        try:
            logging.info(
                'Logging the best model and their metric artifacts'
            )
            with mlflow.start_run():
                f1score=classification_metric.f1score
                recall_score=classification_metric.recall_score
                precision_score=classification_metric.precision_score

                mlflow.log_metric("f1_score",f1score)
                mlflow.log_metric("recall_score",recall_score)
                mlflow.log_metric("precision_score",precision_score)

                mlflow.sklearn.log_model(best_model,'FraudModel')


        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

        
    def Initiate_ModelTraining(self)->ModelTrainerArtifact:
        ''''Initiate the process of Training model'''

        try:
            logging.info(
                f"Initiating the ModelTrainer Processss.."
            )

            train_arr=self.utils.load_numpy_array(filepath=self.data_transformation_artifact.Transformed_Trainarr)
            test_arr=self.utils.load_numpy_array(filepath=self.data_transformation_artifact.Transformed_Testarr)

            if train_arr and test_arr:
                logging.info("load the train and test arary")

                X_train,Y_train,X_test,Y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
                logging.info(
                    f" Got X_train,Y_train,X_test,Y_test from train_arr,test_arr of shape X_train {X_train.shape}, Y_train {Y_train.shape},X_test {X_test.shape},Y_test {Y_test.shape}"
                )

                #Train the model and get best model
                best_model_name, best_model_object,best_model_score,y_test_pred,y_train_pred=self.get_best_model(x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test)
                logging.info(
                    f"Trained th model and Got our best model {best_model_name} and accuracy {best_model_score}"
                )

                if best_model_score>EXPECTED_ACCURACY:
                    logging.info(
                        f"Achieve best score from out expectation {best_model_score} and Expectation {EXPECTED_ACCURACY}")
                    
                    Train_metric=Metric(y_true=Y_train,y_pred=y_train_pred)
                    Test_metric=Metric(y_true=Y_test,y_pred=y_test_pred)
                    self.Track_mlflow(best_model=best_model_name,classification_metric=Train_metric)

                    #load the preprocessor
                    preprocessor=self.utils.load_object(filepath=self.data_transformation_artifact.Preprocessor_Path)
                    best_model=FraudModel(preprocessor=preprocessor,model=best_model_object)

                    os.makedirs(os.path.dirname(self.model_trainer_config.model_path),exist_ok=True)

                    self.utils.save_object(filepath=self.model_trainer_config.model_path,obj=best_model)

                    model_trainer_artifact=ModelTrainerArtifact(
                        BestModelPath=self.model_trainer_config.model_path,
                        train_metric_artifact=Train_metric,
                        test_metric_artifact=Test_metric

                    )

                    return model_trainer_artifact

                else:
                    logging.info(
                        f"Accuracy is less then our Expectation {EXPECTED_ACCURACY} and our model accuracy {best_model_score}"

                    )
            logging.info(
                f"Train_arr and Test_arr are empty ,Pls troubleshoot it..."
            )





        except Exception as e:
            raise CreditCradFraudDetection(e,sys)


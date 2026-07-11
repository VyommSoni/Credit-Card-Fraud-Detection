import os
import sys
from pandas import DataFrame
import pandas as pd
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from src.Constant.TrainingPipeline_Constant import *
from src.entity.Artifact_entity import DataIngestionArtifact,DataTransformationArtifact,DataValidationArtifact,ModelEvaluationArtifact,ModelTrainerArtifact
from src.entity.Config_entity import ModelEvaluationConfig
from src.Utility.Utilis.Utility import  Utils
from src.Components.ML.S3_estimator import FraudDetecionModel
from typing import Optional
from sklearn.metrics import f1_score
from src.Components.ML.Metric import Metric


class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                    model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact):
            
            self.modelevaluationconfig=model_eval_config
            self.utils=Utils()
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
    
    
    def get_best_production_model(self)->Optional[FraudDetecionModel]:
    
            try:
                bucket_name=self.modelevaluationconfig.bucket_name
                s3_model_key_path=self.modelevaluationconfig.s3_model_key_path
                fraud_detection_model=FraudDetecionModel(
                    bucket_name=bucket_name,
                    model_path=s3_model_key_path
                )
    
                if fraud_detection_model.is_model_present(model_path=s3_model_key_path):
                    return fraud_detection_model
                return None
            except Exception as e:
                raise CreditCradFraudDetection(e,sys)
            
    def read_data(self,filepath:str)->DataFrame:
            try:
                return pd.read_csv(filepath)
            except Exception as e:
                raise CreditCradFraudDetection(e,sys)
            
    def Initiate_ModelEvaluation(self)->ModelEvaluationArtifact:
            '''Initiate Model Evaluation'''
    
            try:
                logging.info(
                     f"Intitaing Model Evaluation")
                
                test_df=self.read_data(filepath=self.data_ingestion_artifact.TEST_FILE_PATH)
    
                if not test_df.empty:
                     X_test,Y_test=test_df.drop(columns=COLUMNS_TO_DROP),test_df[TARGET_COLUMN]
    
                     trained_model=self.utils.load_object(filepath=self.model_trainer_artifact.BestModelPath) #load trained model on your local system
    
                     if trained_model is None:
                         raise Exception("Model could not be loaded...")
    
                     y_pred=trained_model.predict(X_test)
    
                     trained_model_f1_score=f1_score(Y_test,y_pred)
                     production_model_f1_score=None
                     production_model_artifact=None
                     best_model=self.get_best_production_model()
    
                     if best_model is not None:
                         y_hat=best_model.Predict(X_test)
                         production_model_f1_score=f1_score(Y_test,y_hat)
                         production_model_artifact=Metric(y_true=Y_test,y_pred=y_hat)
    
                     tmp_best_model_score=0 if production_model_f1_score is None else production_model_f1_score
    
                     model_evaluation_artifact=ModelEvaluationArtifact(
                         is_model_accepted=trained_model_f1_score>tmp_best_model_score,
                         trained_model_path=self.model_trainer_artifact.BestModelPath,
                         best_model=self.model_trainer_artifact.BestModelPath,
                         best_model_artifact=production_model_artifact
    
                         
                     )
    
                     logging.info(
                         f"Model evaluation artifacts {model_evaluation_artifact} "
                     )
    
                     return model_evaluation_artifact
                #Test Git
                
            except Exception as e:
                raise CreditCradFraudDetection(e,sys)
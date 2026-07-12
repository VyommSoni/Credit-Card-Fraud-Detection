import os
import sys
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from pandas import DataFrame
import pandas as pd
from src.entity.Artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelEvaluationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact, ModelPusherArtifact
from src.entity.Config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelEvaluationConfig, ModelPusherConfig, ModelTrainerConfig, TrainingPipelineConfig
from src.Components.DataIngestion.DataIngestion import DataIngestion
from src.Components.DataValidation.DataValidation import DataValidation
from src.Components.DataTransformation.DataTransformation import DataTransformation
from src.Components.Model_Building.ModelBuilding import ModelTrainer
from src.Components.ModelEvaluation.ModelEvaluation import ModelEvaluation
from src.Components.ModelPusher.ModelPusher import ModelPusher
from src.Constant.TrainingPipeline_Constant import *


class TrainingPipeline:
    def __init__(self):
        self.trainingpipelineconfig = TrainingPipelineConfig()

    def read_data(self,filepath:str)->DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
    

    def Initiate_Data_Ingestion(self):
        """Initiate DataIngestion"""
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.trainingpipelineconfig)
            logging.info("Starting DataIngestion..")

            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            artifact = data_ingestion.Initiate_DataIngestion()

            logging.info(f"DataIngestion completed and their artifact is {artifact}")
            return artifact
        
        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

    def Inititiate_Data_Validation(self, ingestion_artifact: DataIngestionArtifact):
        """Initiate DataValidation"""
        try:
            self.data_validation_config = DataValidationConfig(training_pipeline_config=self.trainingpipelineconfig)
            logging.info("Starting Data Validation..")

            validation = DataValidation(data_artifact=ingestion_artifact, datavalidation_config=self.data_validation_config)
            validationartifact = validation.Initiate_DataValidation()

            logging.info(f"Validation Completed and their artifact is {validationartifact}")
            return validationartifact
        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

    def Intitiate_Data_Transformation(self, validation_artifact: DataValidationArtifact):
        """Initiate DataTransformation"""
        try:
            self.data_transformation_config = DataTransformationConfig(training_pipeline_config=self.trainingpipelineconfig)
            logging.info("Starting the Transformation process")

            transformation = DataTransformation(data_transformation_config=self.data_transformation_config, data_validation_artifact=validation_artifact)
            if validation_artifact.Validation_status:
             transformation_artifact = transformation.Initiate_DataTransformation()
            else:
                raise Exception('Validation status is not True cant move ahead for transformation...')

            logging.info(f"Transformation Completed and their artifact is {transformation_artifact}")
            return transformation_artifact

        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

    def Initiate_Model_Trainer(self, transformation_artifact: DataTransformationArtifact,ingestion_artifact:DataIngestionArtifact):
        """Initiate model trainer"""
        try:
            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.trainingpipelineconfig)
            logging.info("Starting the ModelTraining Process")

            # FIX: Switched from self.trainingpipelineconfig to self.model_trainer_config
            model_trainer = ModelTrainer(data_transformation_artifact=transformation_artifact, model_trainer_config=self.model_trainer_config)
            X_train_raw=self.read_data(filepath=ingestion_artifact.TRAIN_FILE_PATH)
            X_test_raw=self.read_data(filepath=ingestion_artifact.TEST_FILE_PATH)
            modeltrainer_artifact = model_trainer.Initiate_ModelTraining(X_train_raw=X_train_raw,X_test_raw=X_test_raw)
            
            logging.info(f"Model Training Completed and their artifact is {modeltrainer_artifact}")
            return modeltrainer_artifact

        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

    def Initiate_Model_Evaluation(self, ingestion_artifact: DataIngestionArtifact, trainer_artifact: ModelTrainerArtifact, transformation_artifact: DataTransformationArtifact):
        """Initiate Model Evaluation"""
        try:
            self.model_eval_config = ModelEvaluationConfig(bucket_name=BUCKET_NAME, s3_model_key_path=MODEL_PATH)
            logging.info("Starting Model Evaluation..")

            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_eval_config,
                data_ingestion_artifact=ingestion_artifact,
                model_trainer_artifact=trainer_artifact,
                data_transformation_artifact=transformation_artifact
            )

            eval_artifact = model_evaluation.Initiate_ModelEvaluation()
            logging.info(f"Model Evaluation Done, and their artifact is {eval_artifact}")
            return eval_artifact

        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

    def Initiate_Model_Puhser(self, trainer_artifact: ModelTrainerArtifact):
        """Initiate Model Pusher"""
        try:
            self.model_pusher_config = ModelPusherConfig(bucket_name=BUCKET_NAME, s3_model_key_path=MODEL_PATH)
            logging.info("Starting the ModelPusher to AWS bucket...")

            model_pusher = ModelPusher(model_trainer_artifact=trainer_artifact, model_pusher_config=self.model_pusher_config)
            pusher_artifact = model_pusher.Initiate_ModelPusher()

            logging.info("Done model pushing to s3")
            return pusher_artifact
        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

    def Run_Pipeline(self):
        """Trigger Entire Pipeline"""
        try:
            data_ingestion_artifact = self.Initiate_Data_Ingestion()
            data_validation_artifact = self.Inititiate_Data_Validation(ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.Intitiate_Data_Transformation(validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.Initiate_Model_Trainer(transformation_artifact=data_transformation_artifact,ingestion_artifact=data_ingestion_artifact)
            
            model_eval_artifact = self.Initiate_Model_Evaluation(
                ingestion_artifact=data_ingestion_artifact,
                trainer_artifact=model_trainer_artifact,
                transformation_artifact=data_transformation_artifact
            )
            
            model_pusher_artifact = None
            # Conditional Deployment Check: Only push to S3 if model is accepted!
            if model_eval_artifact.is_model_accepted:
                model_pusher_artifact = self.Initiate_Model_Puhser(trainer_artifact=model_trainer_artifact)
            else:
                logging.info("Trained model rejected. Current production model outperforms it.")

            return model_trainer_artifact, model_eval_artifact, model_pusher_artifact
        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

if __name__=="__main__":
    TP=TrainingPipeline()
    TP.Run_Pipeline()

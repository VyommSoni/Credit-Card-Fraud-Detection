import os
import sys
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from src.Constant.TrainingPipeline_Constant import *
import pandas as  pd
from src.entity.Artifact_entity import ModelPusherArtifact,ModelTrainerArtifact
from src.entity.Config_entity import ModelPusherConfig
from src.Infrastructure.AWS import SimpleStorageService
from src.Components.ML.S3_estimator import FraudDetecionModel
from src.Utility.Utilis.Utility import Utils



class ModelPusher:
    def __init__(self,model_trainer_artifact:ModelTrainerArtifact,model_pusher_config:ModelPusherConfig):
        self.utils=Utils()
        self.s3=SimpleStorageService()
        self.model_trainer_artifact=model_trainer_artifact
        self.model_pusher_config=model_pusher_config
        self.s3_estimator=FraudDetecionModel(
            bucket_name=self.model_pusher_config.bucket_name,
            model_path=self.model_pusher_config.s3_model_key_path
        )

    def Initiate_ModelPusher(self)->ModelPusherArtifact:
        '''Initiating the modelpusherartifact'''

        try:
            logging.info(
                f"Inititating the model pusher to s3 bucket"
            )

            self.s3_estimator.save_model(
                from_file=self.model_trainer_artifact.BestModelPath
            )

            model_pusher_artifact=ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_key_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info("Upload the model on s3 ")

            logging.info(
                f" model Pushed artifacts {model_pusher_artifact}"

            )



            return model_pusher_artifact
        except Exception as e:
            raise  CreditCradFraudDetection(e,sys)

    
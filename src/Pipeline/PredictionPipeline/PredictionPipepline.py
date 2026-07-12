import os
import sys
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from src.Pipeline.TrainingPipeline.TrainingPipeline import *
from src.Constant.PredictionPipeline_Constant import *
from src.Utility.Utilis.Utility import Utils
from src.Infrastructure.AWS import SimpleStorageService
from src.entity.Artifact_entity import ModelTrainerArtifact
from typing import List
from src.Configuration.S3_Connection import S3

class PredictionPipeline:
    def __init__(self):
        self.utils=Utils()
        self.s3=SimpleStorageService()


    def Download_Model_From_S3(self):
        '''It will download model from s3'''
        try:
            logging.info(
                f"Download model from AWS cloud to do Prediction..."
            )

            self.s3.load_model(bucket_name=BUCKET_NAME,model_name=MODEL_NAME)

            logging.info(
                f"Downloaded Model and ready to to prediction"
            )

        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
        
    def Predict(self,input_features:DataFrame)->List:
        '''This will predict the output '''
        try:
            logging.info("Prediction Pipeline : processing the incoming features ")

            self.Download_Model_From_S3()


            #load the model
            model=self.utils.load_object(filepath=best_model_path)

         
            input_features=input_features.drop(columns=COLUMNS_TO_DROP,axis=1)


            logging.info(
                "Predicting on Input Features...."
            )

            Predictions=model.predict(input_features)

            return Predictions

        except Exception as e:
            raise CreditCradFraudDetection(e,sys)


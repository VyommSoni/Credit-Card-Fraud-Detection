from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from src.Constant.TrainingPipeline_Constant import *
from src.Components.ML.estimator import FraudModel
import os
import sys
from src.Constant.TrainingPipeline_Constant import *
import pandas as pd
from src.Configuration.S3_Connection import S3
from src.Infrastructure.AWS import SimpleStorageService

class FraudDetecionModel:

    def __init__(self,bucket_name:str,model_path:str):
        self.bucket_name=bucket_name
        self.model_path=model_path
        self.loaded_model=FraudModel=None
        self.S3=SimpleStorageService()


    def is_model_present(self,model_path):
        '''It will check if the model present in bucket or not'''

        try:
            return self.S3.s3_key_path_available(bucket_name=self.bucket_name,s3_key=model_path)
        except Exception as e :
            raise CreditCradFraudDetection(e,sys)

    def load_model(self)->FraudModel:
        '''Load the model from model path'''

        try:
            return self.S3.load_model(model_name=self.model_path,bucket_name=self.bucket_name)
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

    def save_model(self,from_file,remove:False):
        '''This will save your model to model path'''

        try:
            self.S3.upload_file(
                from_filename=from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
        
    def Predict(self,X):
        '''Prediction'''
        try:
            if self.loaded_model is None:
                self.loaded_model=self.load_model()
            return self.loaded_model.predict(X)
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

        

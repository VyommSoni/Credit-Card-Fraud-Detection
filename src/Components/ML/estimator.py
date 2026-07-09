from src.Logging.fraud_logging import logging
from src.Exception.fraud_exception import CreditCradFraudDetection
import sys
import os
import pandas as pd
from src.Constant.TrainingPipeline_Constant import *
from sklearn.compose import ColumnTransformer

class FraudModel:
    def __init__(self,preprocessor:ColumnTransformer,model):
        self.model=model
        self.preprocessor=preprocessor

    def predict(self,X):
        try:
            Transformed_X=self.preprocessor.transform(X)
            Predict=self.model.predict(Transformed_X)

            return Predict
        
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
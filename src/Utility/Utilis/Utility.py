import os
import pickle
import numpy as np
import sys
import  yaml
from src.Logging.fraud_logging import logging
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Constant.TrainingPipeline_Constant import *


class Utils:

    def __init__(self):
        
        pass

    def read_ymlfile(self,filepath:str):
        '''
        read the yml file'''

        try:
            with open(filepath,'r') as file:
                content=yaml.safe_load(file)
                return content

        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

        
    def save_numpy_array(self,array:np.array,filepath:str):
        try:
            with open(filepath,'wb') as file:
                content=np.save(file,array)
                return content
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)


    def save_object(self,filepath:str,obj):
        try:
            with open(filepath,'wb') as file:
                content=pickle.dump(obj,file)
                return content
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
        
    def load_object(self,filepath:str):
        try:
            with open(filepath,'rb') as file:
                content=pickle.load(file)
                return content
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

    def load_numpy_array(self,filepath:str):
        try:
            with open(filepath,'rb') as file:
               content= np.load(file)
               return content
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)


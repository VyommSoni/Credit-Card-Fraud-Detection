import os
import sys
from pymongo import MongoClient
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from src.Constant.TrainingPipeline_Constant import *
from dotenv import load_dotenv

load_dotenv()


class MongoDB:
    Client=None

    def __init__(self):
        '''
        This class will create connection to mongod database'''

        try:
            if MongoDB.Client is None:
              MongoDB.Client=MongoClient(os.getenv("MONGO_DB_URL"))
            self.client=MongoDB.Client
            self.database=self.Client[Database_name]
            self.collection=self.database[Collection_name]
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)



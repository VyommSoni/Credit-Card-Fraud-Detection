import os
import sys
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from dotenv import load_dotenv
import pandas as pd
from typing import List
from pandas import DataFrame
from src.Database.MongoDB_Connection import  MongoDB
import numpy as np
from src.Constant.TrainingPipeline_Constant import *

load_dotenv()



class Fraud_Data:
    def __init__(self,database_name:str):
        self.database_name=database_name
        self.client=None

    
    def build_connection(self):
        '''This function will build connection with respect to given database name'''
        try:
            if self.client is None:
                mongo=MongoDB()
                self.client=mongo.client
                self.database=self.client[self.database_name]
                logging.info(
                    "Connection establish successfully"
                    )
            else:
                logging.info(
                    "Already have connection..."
                    )
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
    
    def get_collection_names(self) -> List[str]:
        """Return all collection names."""

        try:

            self.build_connection()

            return self.database.list_collection_names()

        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

    def get_collection_data(self,
                            collection_name: str) -> pd.DataFrame:
        """Fetch one collection as DataFrame."""

        try:

            self.build_connection()

            collection = self.database[collection_name]

            records = list(
                collection.find({}, {"_id": 0})
            )

            dataframe = pd.DataFrame(records)


            logging.info(
                f"Fetched {len(dataframe)} records from {collection_name}"
            )

            return dataframe

        except Exception as e:
            raise CreditCradFraudDetection(e, sys)

    def export_collections_as_dataframe(self):
        """Yield every collection as a DataFrame."""

        try:

            collections = self.get_collection_names()

            for collection_name in collections:

                dataframe = self.get_collection_data(collection_name)

                yield collection_name, dataframe

        except Exception as e:
            raise CreditCradFraudDetection(e, sys)
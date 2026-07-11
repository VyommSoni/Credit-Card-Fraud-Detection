import os
import sys
from  src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from pandas import DataFrame
import pandas as pd


class Rules:
    def __init__(self):
          pass
    def rules(self,row):
        '''In this we define our rules'''

        try:
            logging.info(
                " Creating rules and Tranforming data  "
            )
            score=0

            if row["transaction_hour"] in [0, 1, 2, 3]:
                    score += 30
            
            if row["foreign_transaction"] == 1:
                    score += 20
            
            if row["location_mismatch"] == 1:
                    score += 25
            
            if row["device_trust_score"] < 60:
                    score += 25

            return score
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
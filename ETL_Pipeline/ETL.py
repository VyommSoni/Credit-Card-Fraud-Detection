import sys
import os
import pandas as pd
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from src.Database.MongoDB_Connection import MongoDB
from pandas import DataFrame
from src.Constant.TrainingPipeline_Constant import *


class ETL_Pipeline:

    def Extract(self,FilePath:str)->DataFrame:
        '''This will extract data from my local system'''
        try:
            DataFrame=pd.read_csv(FilePath)
            if not DataFrame.empty:
                logging.info(
                    f"Got the Dataframe and their size is {DataFrame.shape}"
                    )
                return DataFrame
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

    def Transfrom_Data(self,dataframe:DataFrame)->DataFrame:
        '''
        This will be going to transfrom data '''

        try:
            null_values=dataframe.isnull().sum()

            if null_values.sum()>0:
                dataframe=dataframe.dropna()
            logging.info(
                f"Dataframe doesn't contain null values at all {dataframe.isnull().sum()}")
            return dataframe
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

    def Load_data(self,dataframe:DataFrame):
        '''Save the transform data into mongodb database'''

        try:
            Connection=MongoDB() # connection url

            logging.info(
                "Starting the procedure for storing the data into database..")
            

            records=dataframe.to_dict(orient='records')

            Database=Connection.database # database name

            Collection=Connection.collection #Collection  name

            Collection.insert_many(records)

            logging.info(
                f"Inserted data of len {len(records)}"
                )

            return records
        
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

    def Initiate_Pipeline(self):
        '''Run the Entire the process'''
        try:
            ETL=ETL_Pipeline()
            Data=ETL.Extract(FilePath=Data_FilePath)
            Transfrom_Data=ETL.Transfrom_Data(dataframe=Data)
            Load_Data=ETL.Load_data(dataframe=Transfrom_Data)
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)


if __name__=="__main__":
    etl=ETL_Pipeline()
    etl.Initiate_Pipeline()







import os
import sys
import pandas as pd
from pandas import DataFrame
from src.entity.Artifact_entity import DataIngestionArtifact
from src.entity.Config_entity import DataIngestionConfig,TrainingPipelineConfig
from src.DataAccess.Fraud_Data.Fraud_data import Fraud_Data
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from typing import Tuple
from src.Constant.TrainingPipeline_Constant import *
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config=data_ingestion_config

    def Split_Data(self,dataframe:DataFrame)->Tuple[DataFrame,DataFrame]:
        '''This will split the data into train and test'''
        try:
            if not dataframe.empty:
                logging.info('Splitting data into train and test ')
                train,test=train_test_split(dataframe,test_size=TEST_SIZE,random_state=42)
            return train,test
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

    def Initiate_DataIngestion(self)->DataIngestionArtifact:
        '''
        Intitate the process of DataIngestion'''

        try:
            fraud_data=Fraud_Data(Database_name)

            logging.info(
                "Feteching data from mongodb dataabase..."
                )
            dataframe=None
            for collection_name,data in fraud_data.export_collections_as_dataframe():
                if collection_name==Collection_name:
                    dataframe=data
                    break

            if dataframe is None:
                logging.info("Unable to fetch the daata from mongodb")

            train,test=self.Split_Data(dataframe=dataframe)

            logging.info(
                f" Creating dir for storing features {self.data_ingestion_config.feature_store_path}"
            )
            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_path),exist_ok=True)

            dataframe.to_csv(self.data_ingestion_config.feature_store_path)

            train.to_csv(self.data_ingestion_config.Train_file_path)
            test.to_csv(self.data_ingestion_config.Test_file_path)


            dataingestion_artifact=DataIngestionArtifact(
                Feature_Store_Path=self.data_ingestion_config.feature_store_path,
                TEST_FILE_PATH=self.data_ingestion_config.Test_file_path,
                TRAIN_FILE_PATH=self.data_ingestion_config.Train_file_path
            )

            logging.info(
                            f" Data ingetion artifacts {dataingestion_artifact}"
                        ) 
            
            return dataingestion_artifact
            
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

if __name__=="__main__":
    data_ingestion=DataIngestion(data_ingestion_config=DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig()))
    data_ingestion.Initiate_DataIngestion()
        
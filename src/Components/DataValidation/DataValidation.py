import os
import sys
from src.Constant.TrainingPipeline_Constant import *
from src.entity.Config_entity import DataValidationConfig,DataIngestionConfig
from src.entity.Artifact_entity import DataValidationArtifact
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from src.Utility.Utilis.Utility import Utils
from pandas import DataFrame
import pandas as pd


class DataValidation:

    def __init__(self,dataingestion_config:DataIngestionConfig,datavalidation_config:DataValidationConfig):
        self.dataingestion_config=dataingestion_config
        self.datavalidation_config=datavalidation_config
        self.utils=Utils()


    def check_schema_cols(self,dataframe:DataFrame)->bool:
        '''It will check all schemas of columns '''

        try:
            schema_file_path=self.utils.read_ymlfile(CONIFG_PATH)
            Schema_cols=list(schema_file_path['columns'].keys())
            Dataframe_cols=dataframe.columns.to_list()

            if set(Schema_cols)==set(Dataframe_cols):
                logging.info(
                    f"Schema cols and  Dataframe cols are same...{Schema_cols} and {Dataframe_cols}"
                )
                return True
            else:
                logging.info(
                    f"missing col are {set(Schema_cols)-set(Dataframe_cols)}"
                )
                return False

        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
        
    def check_missing_values(self,dataframe:DataFrame)->bool:
        '''check the missing values in dataframe'''
        try:
            Dataframe=pd.read_csv(dataframe)
            if not Dataframe.empty:
                if Dataframe.isnull().sum().sum()>0:
                    logging.info(

                        f"Dataframe contain missing values pls trouble shoot it {Dataframe.isnull().sum().sum()}"
                    )
                    return False
                return True
            logging.info(
                f"Can't able to load dataframe"
            )
                
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

    def read_data(self,filepath:str)->DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

    def Initiate_DataValidation(self)->DataValidationArtifact:
        '''Start the Validation process of the data'''

        try:
            logging.info(
                f"Inititating the DataValidation process.."
            )

            Train=self.read_data(filepath=self.dataingestion_config.Train_file_path,index=False)
            Test=self.read_data(filepath=self.dataingestion_config.Test_file_path,index=False)

            Validation_Status=False

            if not Train.empty and not Test.empty:
                if (self.check_missing_values(dataframe=Train) and self.check_missing_values(Test) and self.check_schema_cols(Train) and self.check_schema_cols(Test)):
                    Validation_Status=True
                    logging.info(
                        f"Validated Train and Test columns"
                    )
                    os.makedirs(os.path.dirname(self.datavalidation_config.valid_train_filepath),exist_ok=True)
                    Train.to_csv(self.datavalidation_config.valid_train_filepath)
                    Test.to_csv(self.datavalidation_config.valid_test_filepath)


                else:
                    logging.info(
                        f" Data Validation are failed,Check agfain the data first then again intitiate validation process... "
                    )
                    
                    os.makedirs(os.path.dirname(self.datavalidation_config.invalid_train_filepath),exist_ok=True)
                    Train.to_csv(self.datavalidation_config.invalid_train_filepath,index=False)
                    Test.to_csv(self.datavalidation_config.invalid_test_filepath,index=False)


                datavalidation_artifact=DataValidationArtifact(
                    Validation_status=Validation_Status,
                     Invalid_train_filepath=self.datavalidation_config.invalid_train_filepath,
                     Invalid_test_filepath=self.datavalidation_config.invalid_test_filepath,
                     Valid_train_filepath=self.datavalidation_config.valid_train_filepath,
                     Valid_test_filepath=self.datavalidation_config.invalid_test_filepath

                )
                return datavalidation_artifact

            logging.info(
                f" Train and Test are empty, Train shape{Train.shape} and Test shape {Test.shape}"
            )
                



        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
    

    

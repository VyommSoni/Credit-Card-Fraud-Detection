from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from pandas import DataFrame
import pandas as pd
import os
import sys
from src.entity.Artifact_entity import DataTransformationArtifact,DataValidationArtifact
from src.entity.Config_entity import DataTransformationConfig
from src.Utility.Utilis.Utility import Utils
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np
from src.Constant.TrainingPipeline_Constant import *

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,data_validation_artifact:DataValidationArtifact):
        self.data_transformation_config=data_transformation_config
        self.data_validation_artifact=data_validation_artifact
        self.utils=Utils()

    def read_data(self,filepath:str)->DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

    def get_transformer_object(self,num_cols,discrete_cols)->ColumnTransformer:
        '''This will return preprocessor object to fit on train and test data'''

        try:
            Num_Pipeline=Pipeline(
                [
                    ("Imputer",SimpleImputer(strategy='median')),
                    ("Scaling",StandardScaler())
                ])

            Discrete_Pipeline=Pipeline(
                [
                    ("Imputer",SimpleImputer(strategy='most_frequent'))
                ]
            )

            Preprocessor=ColumnTransformer(
                [
                    ("numerical_cols",Num_Pipeline,num_cols),
                    ("Discrete_cols",Discrete_Pipeline,discrete_cols)
                ]

            )
            return Preprocessor

        except Exception  as e:
            raise CreditCradFraudDetection(e,sys)

    def Initiate_DataTransformation(self)->DataTransformationArtifact:

        '''Initiate the DataTransformation processs'''

        try:

            if self.data_validation_artifact.Validation_status:
              logging.info(
                 "Initializing the Transformation process.."
             )

              Train_data=self.read_data(filepath=self.data_validation_artifact.Valid_train_filepath)
              Test_data=self.read_data(filepath=self.data_validation_artifact.Valid_test_filepath)

              if not  Train_data.empty and not Test_data.empty:
                  logging.info(
                      f"Train data shape {Train_data.shape} , Test data shape {Test_data.shape}"
                  )

                  Train_data=Train_data.drop(columns=COLUMNS_TO_DROP,axis=1)
                  Test_data=Test_data.drop(columns=COLUMNS_TO_DROP,axis=1)

                  logging.info(
                      f" columns After dropped in train {Train_data.columns} and in test {Test_data.columns}"
                  )

                  X_train,Y_train=Train_data.drop(columns=[TARGET_COLUMN],axis=1),Train_data[TARGET_COLUMN]
                  X_test,Y_test=Test_data.drop(columns=[TARGET_COLUMN],axis=1),Test_data[TARGET_COLUMN]

                  logging.info(
                      f" Split data into X_train,Y_train,X_test,Y_test and of shape {X_train.shape} {Y_train.shape} {X_test.shape} {Y_test.shape}"
                  )

                  Numerical_cols=[col for col in X_train.columns if X_train[col].nunique()>=10 ]
                  Discrete_cols=[col for col in X_train.columns if X_train[col].nunique()<10]

                  Preprocessor=self.get_transformer_object(
                      num_cols=Numerical_cols,
                      discrete_cols=Discrete_cols
                  )

                  transformed_X_train,transformed_X_test=Preprocessor.fit_transform(X_train),Preprocessor.transform(X_test)
                  logging.info(
                      f"Transformed num and discrete cols "
                  )

                  os.makedirs(os.path.dirname(self.data_transformation_config.preprocessing_path),exist_ok=True)
                  self.utils.save_object(filepath=self.data_transformation_config.preprocessing_path,obj=Preprocessor)

                  train_arr=np.c_[transformed_X_train,np.array(Y_train)]
                  test_arr=np.c_[transformed_X_test,np.array(Y_test)]

                  self.utils.save_numpy_array(array=train_arr,filepath=self.data_transformation_config.transformed_train)
                  self.utils.save_numpy_array(array=test_arr,filepath=self.data_transformation_config.transformed_test)

                  logging.info(
                      f' Save train and test array in {self.data_transformation_config.transformed_train} and test  {self.data_transformation_config.transformed_test}' 
                  )

                  data_transformation_artifact=DataTransformationArtifact(
                      Preprocessor_Path=self.data_transformation_config.preprocessing_path,
                      Transformed_Trainarr=self.data_transformation_config.transformed_train,
                      Transformed_Testarr=self.data_transformation_config.transformed_test
                  )

                  logging.info(
                      f"Data Transformation artifact {data_transformation_artifact}"
                  )

                  return data_transformation_artifact
              else:
                  logging.info(
                      f"Train data and Test data are empty {Train_data.shape} and {Test_data.shape}"
                  )
            logging.info(
                f" Validation Status is not True ,Trouble Shoot the Validation Pipeline : Validation Status {self.data_validation_artifact.Validation_status}"
            )

        except Exception as e:
            raise CreditCradFraudDetection(e,sys)

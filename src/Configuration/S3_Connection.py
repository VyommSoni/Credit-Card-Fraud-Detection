from src.Logging.fraud_logging import logging
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Constant.TrainingPipeline_Constant import *
import boto3
import sys
import os
from dotenv import load_dotenv

load_dotenv()


class S3:

    resource=None
    S3_client=None

    def __init__(self,Region_name=os.getenv("REGION_NAME")):
        '''It Will create the connection for S3 Bucket'''

        try:
            if S3.resource==None or S3.S3_client==None:
                __aws_access_key=os.getenv("ACCESS_KEY")
                __secret_access_key=os.getenv("SECRET_ACCESS_KEY")

                if __aws_access_key is None:
                    raise Exception(f"Environment Varibale : {__aws_access_key} is not set ")
                if __secret_access_key is None:
                    raise Exception (f"Environment varaible {__secret_access_key} is not set")
                
                S3.S3_client=boto3.client(
                    's3',
                 aws_access_key_id=__aws_access_key,
                 aws_secret_access_key=__secret_access_key,
                 region_name=Region_name)
            
                S3.resource=boto3.resource('s3',
                aws_access_key_id=__aws_access_key,
                                 aws_secret_access_key=__secret_access_key,
                                 region_name=Region_name)

            self.S3_client=S3.S3_client
            self.resource=S3.resource

        except Exception  as e:
            raise CreditCradFraudDetection(e,sys)




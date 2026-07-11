import os
import sys
from pandas import DataFrame
import pandas as pd
import numpy as np
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from src.Constant.TrainingPipeline_Constant import *
from src.entity
from src.Components.ML.S3_estimator import FraudDetecionModel


class ModelEvaluation:

    def __init__(self):
        pass
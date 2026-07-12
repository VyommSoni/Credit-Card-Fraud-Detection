import pandas 
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from src.Exception.fraud_exception import CreditCradFraudDetection
from src.Logging.fraud_logging import logging
from src.Constant.TrainingPipeline_Constant import *
from src.entity.Artifact_entity import ClassificationMetricArtifact
import os
import sys

class Metric:
    def __init__(self):
        pass
    def calculate_metric(self,y_true,y_pred)->ClassificationMetricArtifact:
        try:
            logging.info(
                f'calculating model metrics for evaluating model...'
            )
            f1score=f1_score(y_true,y_pred)
            recallscore=recall_score(y_true,y_pred)
            precisionscore=precision_score(y_true,y_pred)
            accuracy_score_=accuracy_score(y_true,y_pred)


            classification_metric_artifact=ClassificationMetricArtifact(
                f1score=f1score,
                recall_score=recallscore,
                precision_score=precisionscore,
                accuracy_score=accuracy_score_

            )

            return classification_metric_artifact
        except Exception as e:
            raise CreditCradFraudDetection(e,sys)
        
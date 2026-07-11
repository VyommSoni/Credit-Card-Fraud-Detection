from src.Logging.fraud_logging import logging
from src.Exception.fraud_exception import CreditCradFraudDetection
import sys
import os
import numpy as np
import pandas as pd
from src.Buissness_rules.Rules import Rules
from src.Constant.TrainingPipeline_Constant import *
from sklearn.compose import ColumnTransformer

class FraudModel:
    def __init__(self,preprocessor:ColumnTransformer,base_model,meta_model):
        self.base_model=base_model
        self.meta_model=meta_model
        self.preprocessor=preprocessor
        self.rules_engine=Rules()

    def _generate_hybrid_feature(self, X_transformed: np.array, X_raw: pd.DataFrame) -> np.array:
        """Helper to create the combined 60/40 hybrid feature matrix"""
        # Stage 1 Probability out of 100
        lr_proba = self.base_model.predict_proba(X_transformed)[:, 1] * 100
        
        # Business Rules out of 100
        rule_scores = X_raw.apply(self.rules_engine.rules, axis=1).values
        
        # Blended Risk Feature
        hybrid_score = (MODEL_WEIGHTAGE * lr_proba) + (BUISSNESS_RULES_WEIGHTAGE * rule_scores)
        return hybrid_score.reshape(-1, 1)


    def predict(self, X: pd.DataFrame) -> np.array:
        """Used directly by Evaluation and S3 inference pipelines"""
        try:
            transformed_X = self.preprocessor.transform(X)
            hybrid_feature = self._generate_hybrid_feature(transformed_X, X)
            
            # Append the hybrid score alongside original features
            stacked_features = np.hstack((transformed_X, hybrid_feature))
            
            return self.meta_model.predict(stacked_features)
        except Exception as e:
            raise CreditCradFraudDetection(e, sys)
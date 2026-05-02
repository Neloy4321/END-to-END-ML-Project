# =========================
# Estimator (Sleep Project)
# =========================

import sys
from pandas import DataFrame

from sleep_project.exception import CustomException
from sleep_project.logger import logging


# -------- Target Mapping --------
class TargetValueMapping:
    def __init__(self):
        self.No_Condition: int = 0
        self.Sleep_Respiratory_Disorders: int = 1
        self.Health_Issues: int = 2
        self.Mental_Health_Issues: int = 3
        self.Others: int = 4

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping = self._asdict()
        return dict(zip(mapping.values(), mapping.keys()))
    

class SleepModel:

    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame):

        try:
            logging.info("Starting prediction")

            # -------- Load preprocessing info --------
            selected_features = self.preprocessing_object["selected_features"]

            # -------- Ensure numeric --------
            dataframe = dataframe.apply(lambda x: x.astype(float), errors='ignore')

            # -------- Align columns --------
            dataframe = dataframe.reindex(columns=selected_features, fill_value=0)

            # -------- Predict --------
            predictions = self.trained_model_object.predict(dataframe)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    
    
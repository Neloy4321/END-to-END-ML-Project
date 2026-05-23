# =========================
# FINAL ESTIMATOR
# =========================

import sys
import pandas as pd
import numpy as np

from pandas import DataFrame

from sleep_project.exception import CustomException
from sleep_project.logger import logging


# =========================
# TARGET LABEL MAPPING
# =========================
class TargetValueMapping:

    def __init__(self):

        self.No_Condition = 0
        self.Sleep_Respiratory_Disorders = 1
        self.Health_Issues = 2
        self.Mental_Health_Issues = 3
        self.Others = 4

    def _asdict(self):

        return self.__dict__

    def reverse_mapping(self):

        return {v: k for k, v in self._asdict().items()}


# =========================
# MODEL CLASS
# =========================
class SleepModel:

    def __init__(
        self,
        preprocessing_object,
        trained_model_object
    ):

        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    # =========================
    # PREDICT
    # =========================
    def predict(self, dataframe):

        try:

            logging.info(
                "Starting prediction"
            )

            # =========================
            # NUMPY ARRAY CASE
            # =========================
            if isinstance(
                dataframe,
                np.ndarray
            ):

                predictions = (
                    self.trained_model_object
                    .predict(dataframe)
                )

                return predictions

            # =========================
            # LOAD OBJECTS
            # =========================
            selected_features = (
                self.preprocessing_object
                .get(
                    "selected_features",
                    []
                )
            )

            feature_encoders = (
                self.preprocessing_object
                .get(
                    "feature_encoders",
                    {}
                )
            )

            print(
                "\n========== ORIGINAL INPUT =========="
            )
            print(dataframe)

            # =========================
            # CLEAN COLUMNS
            # =========================
            dataframe.columns = (
                dataframe.columns
                .str.strip()
                .str.replace(
                    " )",
                    "",
                    regex=False
                )
            )

            # =========================
            # MULTI LABEL EXPANSION
            # =========================
            multi_cols = [

                'Do you experience any of the following side effects from late sleeping?',

                'What are the main reasons you sleep late?',

                'What strategies do you use to cope with the side effects of late sleeping?'
            ]

            for col in multi_cols:

                if col in dataframe.columns:

                    dummies = (

                        dataframe[col]
                        .astype(str)
                        .str.get_dummies(sep=';')
                    )

                    dataframe = pd.concat(
                        [dataframe, dummies],
                        axis=1
                    )

                    dataframe.drop(
                        col,
                        axis=1,
                        inplace=True
                    )

            # =========================
            # CLEAN DUPLICATE COLUMNS
            # =========================
            dataframe.columns = (
                dataframe.columns
                .str.strip()
                .str.replace(
                    " )",
                    "",
                    regex=False
                )
            )

            dataframe = dataframe.groupby(
                dataframe.columns,
                axis=1
            ).sum()

            print(
                "\n========== AFTER DUMMY EXPANSION =========="
            )
            print(dataframe)

            # =========================
            # APPLY LABEL ENCODERS
            # =========================
            for col, encoder in feature_encoders.items():

                if col in dataframe.columns:

                    dataframe[col] = (
                        dataframe[col]
                        .astype(str)
                    )

                    # unseen handling
                    dataframe[col] = (
                        dataframe[col]
                        .apply(

                            lambda x:
                            x
                            if x in encoder.classes_
                            else encoder.classes_[0]
                        )
                    )

                    dataframe[col] = (
                        encoder.transform(
                            dataframe[col]
                        )
                    )

            print(
                "\n========== AFTER ENCODING =========="
            )
            print(dataframe)

            # =========================
            # ALIGN FEATURES
            # =========================
            dataframe = dataframe.reindex(
                columns=selected_features,
                fill_value=0
            )

            print(
                "\n========== AFTER REINDEX =========="
            )
            print(dataframe)

            # =========================
            # FILL MISSING
            # =========================
            dataframe = dataframe.fillna(0)

            # =========================
            # FORCE FLOAT
            # =========================
            dataframe = dataframe.astype(float)

            print(
                "\n========== FINAL INPUT =========="
            )
            print(dataframe)

            # =========================
            # PREDICT
            # =========================
            predictions = (
                self.trained_model_object
                .predict(dataframe)
            )

            print(
                "\n========== FINAL PREDICTIONS =========="
            )
            print(predictions)

            return predictions

        except Exception as e:

            raise CustomException(e, sys)

    def __repr__(self):

        return (
            f"{type(self.trained_model_object).__name__}()"
        )

    def __str__(self):

        return (
            f"{type(self.trained_model_object).__name__}()"
        )
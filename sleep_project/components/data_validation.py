# =========================
# Data Validation Component (Sleep Project)
# =========================

import sys
import os
import json
import pandas as pd
import numpy as np

from pandas import DataFrame
from typing import Optional

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from sleep_project.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from sleep_project.entity.config_entity import DataValidationConfig
from sleep_project.constants import SCHEMA_FILE_PATH
from sleep_project.logger import logging
from sleep_project.exception import CustomException
from sleep_project.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self.schema_config["columns"])
            logging.info(f"Column count validation status: {status}")
            return status
        except Exception as e:
            raise CustomException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = df.columns

            missing_numerical_columns = []
            missing_categorical_columns = []

            for col in self.schema_config["numerical_columns"]:
                if col not in dataframe_columns:
                    missing_numerical_columns.append(col)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            for col in self.schema_config["categorical_columns"]:
                if col not in dataframe_columns:
                    missing_categorical_columns.append(col)

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return False if missing_numerical_columns or missing_categorical_columns else True

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            profile.calculate(reference_df, current_df)

            report = profile.json()
            json_report = json.loads(report)

            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=json_report
            )

            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation...")

            train_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.trained_file_path
            )
            test_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            validation_error_msg = ""

            # Column count validation
            status = self.validate_number_of_columns(train_df)
            if not status:
                validation_error_msg += "Train dataframe column mismatch. "

            status = self.validate_number_of_columns(test_df)
            if not status:
                validation_error_msg += "Test dataframe column mismatch. "

            # Column existence validation
            status = self.is_column_exist(train_df)
            if not status:
                validation_error_msg += "Missing columns in train dataframe. "

            status = self.is_column_exist(test_df)
            if not status:
                validation_error_msg += "Missing columns in test dataframe. "

            validation_status = len(validation_error_msg) == 0

            # Drift check
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    validation_error_msg = "Data drift detected"
                else:
                    validation_error_msg = "No data drift detected"

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data Validation Artifact: {data_validation_artifact}")

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)
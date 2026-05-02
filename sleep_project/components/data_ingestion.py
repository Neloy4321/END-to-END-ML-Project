# =========================
# Data Ingestion Component (Production Ready)
# =========================

import os
import sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from sleep_project.entity.config_entity import DataIngestionConfig
from sleep_project.entity.artifact_entity import DataIngestionArtifact
from sleep_project.data_access.sleep_project import SleepData
from sleep_project.logger import logging
from sleep_project.exception import CustomException


class DataIngestion:
    """
    Data Ingestion class:
    - MongoDB → DataFrame
    - Save feature store
    - Train/Test split
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from MongoDB...")

            sleep_data = SleepData()
            dataframe = sleep_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            logging.info(f"Data shape: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Saving data into feature store: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe

        except Exception as e:
            raise CustomException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        try:
            logging.info("Splitting data into train and test sets...")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )

            logging.info("Train/Test split completed.")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Data Ingestion started...")

            dataframe = self.export_data_into_feature_store()

            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data Ingestion completed. Artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)
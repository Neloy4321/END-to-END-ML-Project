# =========================
# Training Pipeline (Sleep Project)
# =========================

import sys

from sleep_project.exception import CustomException
from sleep_project.logger import logging
from sleep_project.components.data_ingestion import DataIngestion

from sleep_project.entity.config_entity import DataIngestionConfig
from sleep_project.entity.artifact_entity import DataIngestionArtifact


class TrainingPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Start data ingestion component
        """
        try:
            logging.info("Entered start_data_ingestion method")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed")
            logging.info("Exited start_data_ingestion method")

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self) -> None:
        """
        Run complete pipeline
        """
        try:
            logging.info("Pipeline started")

            data_ingestion_artifact = self.start_data_ingestion()

            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            logging.info("Pipeline completed")

        except Exception as e:
            raise CustomException(e, sys)
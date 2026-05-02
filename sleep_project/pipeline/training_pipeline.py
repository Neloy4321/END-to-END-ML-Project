# =========================
# Training Pipeline (Sleep Project)
# =========================

import sys

from sleep_project.exception import CustomException
from sleep_project.logger import logging
from sleep_project.components.data_ingestion import DataIngestion
from sleep_project.components.data_validation import DataValidation
from sleep_project.components.data_transformation import DataTransformation
from sleep_project.components.model_trainer import ModelTrainer

from sleep_project.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from sleep_project.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact


class TrainingPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        
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
        
    
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:

        try:
            logging.info("Starting data validation")

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)
        
        
        
    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:

        try:
            logging.info("Starting data transformation")

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config
            )

            data_transformation_artifact = data_transformation.initiate_data_transformation()

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
        
        
        
    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
        ) -> ModelTrainerArtifact:
        """
        This method is responsible for starting model training
        """
        try:
            model_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e, sys)    
        
        
        
    def run_pipeline(self) -> None:
        """
        Run complete pipeline
        """
        try:
            logging.info("Pipeline started")

            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )

            # ===== ADD THIS =====
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            # ====================

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            logging.info("Pipeline completed")

        except Exception as e:
            raise CustomException(e, sys)
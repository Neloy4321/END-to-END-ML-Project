import sys

from sleep_project.exception import CustomException
from sleep_project.logger import logging

from sleep_project.components.data_ingestion import DataIngestion
from sleep_project.components.data_validation import DataValidation
from sleep_project.components.data_transformation import DataTransformation
from sleep_project.components.model_trainer import ModelTrainer
from sleep_project.components.model_evaluation import ModelEvaluation
from sleep_project.components.model_pusher import ModelPusher

from sleep_project.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from sleep_project.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)


class TrainingPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    # ---------------- INGESTION ----------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Entered data ingestion")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            return data_ingestion.initiate_data_ingestion()

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- VALIDATION ----------------
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )

            return data_validation.initiate_data_validation()

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- TRANSFORMATION ----------------
    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:

        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config
            )

            return data_transformation.initiate_data_transformation()

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- TRAINER ----------------
    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:

        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )

            return model_trainer.initiate_model_trainer()

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- EVALUATION (FIXED) ----------------
    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:

        try:
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_artifact=data_transformation_artifact,  # ✅ FIX
                model_trainer_artifact=model_trainer_artifact
            )

            return model_evaluation.initiate_model_evaluation()

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- PUSHER ----------------
    def start_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:

        try:
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config
            )

            return model_pusher.initiate_model_pusher()

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- RUN PIPELINE ----------------
    def run_pipeline(self) -> None:

        try:
            logging.info("Pipeline started")

            # 1. Ingestion
            data_ingestion_artifact = self.start_data_ingestion()

            # 2. Validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            # 3. Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )

            # 4. Training
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            # 5. Evaluation (FIXED)
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_artifact=data_transformation_artifact,  # ✅ FIX
                model_trainer_artifact=model_trainer_artifact
            )

            # 6. Check Acceptance
            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model rejected. Pipeline stopped.")
                return

            # 7. Pusher
            self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )

            logging.info("Pipeline completed successfully")

        except Exception as e:
            raise CustomException(e, sys)
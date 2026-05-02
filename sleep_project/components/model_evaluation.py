import sys
from typing import Optional
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score

from sleep_project.entity.config_entity import ModelEvaluationConfig
from sleep_project.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact
)

from sleep_project.entity.s3_estimator import S3Estimator
from sleep_project.exception import CustomException
from sleep_project.logger import logging
from sleep_project.utils.main_utils import load_numpy_array_data


# ---------------- RESPONSE ----------------
@dataclass
class EvaluateModelResponse:
    trained_model_accuracy: float
    best_model_accuracy: float
    is_model_accepted: bool
    difference: float


# ---------------- MAIN CLASS ----------------
class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- LOAD OLD MODEL FROM S3 ----------------
    def get_best_model(self) -> Optional[S3Estimator]:
        try:
            estimator = S3Estimator(
                bucket_name=self.model_eval_config.bucket_name,
                model_path=self.model_eval_config.s3_model_key_path
            )

            if estimator.is_model_present(self.model_eval_config.s3_model_key_path):
                return estimator

            return None

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- EVALUATE ----------------
    def evaluate_model(self) -> EvaluateModelResponse:

        try:
            logging.info("Starting model evaluation")

            # ✅ USE TRANSFORMED TEST DATA (CRITICAL FIX)
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # ---------------- NEW MODEL ----------------
            trained_model_accuracy = self.model_trainer_artifact.metric_artifact.accuracy

            # ---------------- OLD MODEL (S3) ----------------
            best_model = self.get_best_model()
            best_model_accuracy = None

            if best_model is not None:
                y_pred_old = best_model.predict(X_test)
                best_model_accuracy = accuracy_score(y_test, y_pred_old)

            # ---------------- COMPARE ----------------
            base_score = 0 if best_model_accuracy is None else best_model_accuracy

            is_accepted = trained_model_accuracy > base_score
            diff = trained_model_accuracy - base_score

            result = EvaluateModelResponse(
                trained_model_accuracy=trained_model_accuracy,
                best_model_accuracy=best_model_accuracy,
                is_model_accepted=is_accepted,
                difference=diff
            )

            logging.info(f"Evaluation Result: {result}")

            return result

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- INIT ----------------
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:
            response = self.evaluate_model()

            artifact = ModelEvaluationArtifact(
                is_model_accepted=response.is_model_accepted,
                changed_accuracy=response.difference,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )

            logging.info(f"Model Evaluation Artifact: {artifact}")

            return artifact

        except Exception as e:
            raise CustomException(e, sys)
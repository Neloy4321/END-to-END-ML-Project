import sys

from sleep_project.cloud_storage.aws_storage import SimpleStorageService
from sleep_project.exception import CustomException
from sleep_project.logger import logging

from sleep_project.entity.artifact_entity import (
    ModelPusherArtifact,
    ModelEvaluationArtifact
)

from sleep_project.entity.config_entity import ModelPusherConfig
from sleep_project.entity.s3_estimator import S3Estimator


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig
    ):
        """
        Upload accepted model to S3
        """

        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

        self.estimator = S3Estimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path
        )

    # ---------------- MAIN ----------------
    def initiate_model_pusher(self) -> ModelPusherArtifact:

        logging.info("Entered Model Pusher")

        try:
            # -------- CHECK ACCEPTANCE --------
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted. Skipping upload.")
                
                return ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=None
                )

            # -------- UPLOAD --------
            logging.info("Uploading model to S3...")

            self.estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_path,
                remove=False
            )

            logging.info("Model uploaded successfully")

            # -------- ARTIFACT --------
            artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info(f"Model Pusher Artifact: {artifact}")

            return artifact

        except Exception as e:
            raise CustomException(e, sys)
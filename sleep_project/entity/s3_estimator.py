import sys
from pandas import DataFrame

from sleep_project.cloud_storage.aws_storage import SimpleStorageService
from sleep_project.exception import CustomException
from sleep_project.entity.estimator import SleepModel


class S3Estimator:
    """
    S3 থেকে model load করে prediction করার জন্য ব্যবহার হবে
    """

    def __init__(self, bucket_name: str, model_path: str):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: SleepModel = None

    # ---------------- CHECK MODEL ----------------
    def is_model_present(self, model_path: str) -> bool:
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=model_path
            )
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- LOAD MODEL ----------------
    def load_model(self) -> SleepModel:
        try:
            return self.s3.load_model(
                model_name=self.model_path,
                bucket_name=self.bucket_name
            )
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- SAVE MODEL ----------------
    def save_model(self, from_file: str, remove: bool = False) -> None:
        try:
            self.s3.upload_file(
                from_path=from_file,
                to_path=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- PREDICT ----------------
    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            return self.loaded_model.predict(dataframe=dataframe)

        except Exception as e:
            raise CustomException(e, sys)
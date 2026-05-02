import os
import sys
import pickle
from io import StringIO
from typing import Union, List

from pandas import DataFrame, read_csv
from botocore.exceptions import ClientError

from sleep_project.configuration.aws_connection import S3Client
from sleep_project.logger import logging
from sleep_project.exception import CustomException


class SimpleStorageService:

    def __init__(self):
        s3 = S3Client()
        self.s3_resource = s3.s3_resource
        self.s3_client = s3.s3_client

    # ---------------- CHECK PATH ----------------
    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)
            objects = [obj for obj in bucket.objects.filter(Prefix=s3_key)]
            return len(objects) > 0
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- READ OBJECT ----------------
    @staticmethod
    def read_object(object_name, decode=True, make_readable=False):
        try:
            data = object_name.get()["Body"].read()
            if decode:
                data = data.decode()

            return StringIO(data) if make_readable else data

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- GET BUCKET ----------------
    def get_bucket(self, bucket_name):
        try:
            return self.s3_resource.Bucket(bucket_name)
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- GET FILE ----------------
    def get_file_object(self, filename, bucket_name):
        try:
            bucket = self.get_bucket(bucket_name)
            objects = [obj for obj in bucket.objects.filter(Prefix=filename)]
            return objects[0] if len(objects) == 1 else objects
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- LOAD MODEL ----------------
    def load_model(self, model_name, bucket_name, model_dir=None):
        try:
            key = model_name if model_dir is None else f"{model_dir}/{model_name}"
            obj = self.get_file_object(key, bucket_name)
            model_bytes = self.read_object(obj, decode=False)
            return pickle.loads(model_bytes)
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- CREATE FOLDER ----------------
    def create_folder(self, folder_name, bucket_name):
        try:
            self.s3_resource.Object(bucket_name, folder_name).load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_name + "/")
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- UPLOAD FILE ----------------
    def upload_file(self, from_path, to_path, bucket_name, remove=True):
        try:
            self.s3_client.upload_file(from_path, bucket_name, to_path)

            if remove:
                os.remove(from_path)

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- UPLOAD DF ----------------
    def upload_df_as_csv(self, df: DataFrame, local_file, s3_file, bucket_name):
        try:
            df.to_csv(local_file, index=False)
            self.upload_file(local_file, s3_file, bucket_name)
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- READ DF ----------------
    def get_df_from_object(self, obj) -> DataFrame:
        try:
            content = self.read_object(obj, make_readable=True)
            return read_csv(content)
        except Exception as e:
            raise CustomException(e, sys)

    def read_csv(self, filename, bucket_name):
        try:
            obj = self.get_file_object(filename, bucket_name)
            return self.get_df_from_object(obj)
        except Exception as e:
            raise CustomException(e, sys)
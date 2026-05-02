import os
import boto3

from sleep_project.constants import (
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    AWS_ACCESS_KEY_ID_ENV_KEY,
    REGION_NAME
)

from sleep_project.exception import CustomException
import sys


class S3Client:
    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):
        """
        Create S3 client & resource using env variables
        """

        try:
            if S3Client.s3_client is None or S3Client.s3_resource is None:

                access_key = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
                secret_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

                if access_key is None:
                    raise Exception(f"{AWS_ACCESS_KEY_ID_ENV_KEY} not set")

                if secret_key is None:
                    raise Exception(f"{AWS_SECRET_ACCESS_KEY_ENV_KEY} not set")

                S3Client.s3_resource = boto3.resource(
                    's3',
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region_name
                )

                S3Client.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region_name
                )

            self.s3_resource = S3Client.s3_resource
            self.s3_client = S3Client.s3_client

        except Exception as e:
            raise CustomException(e, sys)
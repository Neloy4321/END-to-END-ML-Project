# =========================
# Data Access Layer (Mongo → DataFrame)
# =========================

import pandas as pd
import sys
import numpy as np
from typing import Optional

from sleep_project.configuration.mongo_db_connection import MongoDBClient
from sleep_project.constants import DATABASE_NAME


class SleepData:
    """
    This class helps to export entire MongoDB collection as pandas DataFrame
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise Exception(e)

    def export_collection_as_dataframe(
        self,
        collection_name: str,
        database_name: Optional[str] = None
    ) -> pd.DataFrame:

        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            # Drop MongoDB default id
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)

            # Replace "na" with np.nan
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise Exception(e)
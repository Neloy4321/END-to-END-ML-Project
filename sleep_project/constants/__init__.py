# =========================
# Constants (Global Config)
# =========================

import os
from datetime import date

# -------- MongoDB --------
DATABASE_NAME = "sleep_data"
COLLECTION_NAME = "sleep_data_collection"
MONGODB_URL_KEY = "MONGODB_URL"

# -------- Pipeline --------
PIPELINE_NAME: str = "sleep_pipeline"
ARTIFACT_DIR: str = "artifact"

# -------- Files --------
FILE_NAME: str = "sleep_data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

MODEL_FILE_NAME = "model.pkl"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

# -------- Target --------
TARGET_COLUMN = "Medical Condition Category"

# -------- Misc --------
CURRENT_YEAR = date.today().year

# -------- Schema Path --------
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")



# =========================
# Data Ingestion Constants
# =========================

DATA_INGESTION_COLLECTION_NAME: str = "sleep_data_collection"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# =========================
# Data Validation Constants (Sleep Project)
# =========================

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


# =========================
# Data Transformation Constants (Sleep Project)
# =========================

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


# =========================
# Model Trainer Constants (Sleep Project)
# =========================

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

# Expected minimum accuracy (tune later if needed)
MODEL_TRAINER_EXPECTED_SCORE: float = 0.2

# Path to model config (yaml)
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
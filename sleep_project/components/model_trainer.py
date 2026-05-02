import sys
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

import importlib

from sleep_project.exception import CustomException
from sleep_project.logger import logging
from sleep_project.utils.main_utils import (
    load_numpy_array_data,
    read_yaml_file,
    load_object,
    save_object
)

from sleep_project.entity.config_entity import ModelTrainerConfig
from sleep_project.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)

from sleep_project.entity.estimator import SleepModel


class ModelTrainer:

    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    # -------- Load models dynamically --------
    def load_models(self):
        config = read_yaml_file(self.model_trainer_config.model_config_file_path)

        models = {}
        params = {}

        for key, model_info in config["model_selection"].items():
            module = importlib.import_module(model_info["module"])
            model_class = getattr(module, model_info["class"])

            models[key] = model_class(**model_info["params"])
            params[key] = model_info["search_param_grid"]

        return models, params

    # -------- Train + Evaluate --------
    def get_best_model(self, train: np.array, test: np.array) -> Tuple[object, ClassificationMetricArtifact]:

        try:
            X_train, y_train = train[:, :-1], train[:, -1]
            X_test, y_test = test[:, :-1], test[:, -1]

            models, param_grid = self.load_models()

            best_model = None
            best_score = -1
            best_metric = None

            for key in models:
                print(f"\n Training model: {key}")
                logging.info(f"Training model: {key}")

                grid = GridSearchCV(
                    models[key],
                    param_grid[key],
                    cv=3,
                    verbose=2,
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                model = grid.best_estimator_
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                print(f" {key} Accuracy: {acc}")
                logging.info(f"{key} Accuracy: {acc}")

                if acc > best_score:
                    best_score = acc
                    best_model = model

                    best_metric = ClassificationMetricArtifact(
                        accuracy=acc,
                        f1_score=f1_score(y_test, y_pred, average='weighted'),
                        precision=precision_score(y_test, y_pred, average='weighted'),
                        recall=recall_score(y_test, y_pred, average='weighted')
                    )

            print(f"\n Best Model Accuracy: {best_score}")
            logging.info(f"Best Model Accuracy: {best_score}")

            return best_model, best_metric

        except Exception as e:
            raise CustomException(e, sys)

    # -------- Main Trainer --------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:
            print("\n Starting Model Training...\n")
            logging.info("Starting Model Training")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            best_model, metric_artifact = self.get_best_model(train_arr, test_arr)

            if metric_artifact.accuracy < self.model_trainer_config.expected_accuracy:
                print(" Model below expected accuracy, but continuing...")
                logging.warning("Model below expected accuracy")

            preprocessing_obj = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # -------- Wrap model --------
            sleep_model = SleepModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model
            )

            save_object(self.model_trainer_config.trained_model_file_path, sleep_model)

            print("\n Model saved successfully!")
            logging.info("Model saved successfully")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise CustomException(e, sys)
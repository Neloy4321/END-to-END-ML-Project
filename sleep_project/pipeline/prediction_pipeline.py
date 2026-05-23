import os
import sys
import pickle

import pandas as pd
from pandas import DataFrame

from sleep_project.exception import CustomException
from sleep_project.logger import logging


# =========================
# DATA CLASS
# =========================
class SleepData:

    def __init__(
        self,
        age,
        weight,
        height,
        gender,
        occupation,
        bed_time,
        wake_time,
        sleep_hours,
        difficulty_sleep,
        breathing_problem,
        restless_legs,
        concentration_problem,
        sleep_environment,
        sleep_reason,
        medical_condition,
        coping_strategy
    ):

        try:

            self.age = age
            self.weight = weight
            self.height = height
            self.gender = gender
            self.occupation = occupation
            self.bed_time = bed_time
            self.wake_time = wake_time
            self.sleep_hours = sleep_hours
            self.difficulty_sleep = difficulty_sleep
            self.breathing_problem = breathing_problem
            self.restless_legs = restless_legs
            self.concentration_problem = concentration_problem
            self.sleep_environment = sleep_environment
            self.sleep_reason = sleep_reason
            self.medical_condition = medical_condition
            self.coping_strategy = coping_strategy

        except Exception as e:
            raise CustomException(e, sys)

    # =========================
    # SAFE FLOAT
    # =========================
    def _to_float(self, x, default=0.0):

        try:
            return float(x)

        except:
            return default

    # =========================
    # CREATE RAW DATAFRAME
    # =========================
    def get_sleep_input_data_frame(self) -> DataFrame:

        try:

            # =========================
            # LOW SLEEP CHECK
            # =========================
            low_sleep = self.sleep_hours in [
                "Less than 4 hours",
                "4-6 hours"
            ]

            # =========================
            # SIDE EFFECTS
            # =========================
            if low_sleep:
                side_effects = (
                    "Fatigue ;Difficulty concentrating"
                    )
            else:
                side_effects = "None"

            # =========================
            # RAW DATAFRAME
            # =========================
            input_dict = {

                "Your Age": [
                    self.age
                ],

                "What is your weight": [
                    self._to_float(self.weight)
                ],

                "Your Height": [
                    self._to_float(self.height)
                ],

                "What is your gender?": [
                    self.gender
                ],

                "What is your occupation?": [
                    self.occupation
                ],

                "What time do you usually go to bed?": [
                    self.bed_time
                ],

                "What time do you usually wake up on working days?": [
                    self.wake_time
                ],

                "What time do you usually go to bed on weekends?": [
                    self.bed_time
                ],

                "What time do you usually wake up on weekends?": [
                    self.wake_time
                ],

                "How long does it take you to fall asleep after going to bed?": [
                    "15-30 minutes"
                ],

                "How many hours of sleep do you get on average per night?": [
                    self.sleep_hours
                ],

                "What are the main reasons you sleep late?": [
                    self.sleep_reason
                ],

                "Do you have difficulty falling asleep?": [
                    self.difficulty_sleep
                ],

                "Do you experience breathing difficulties while sleeping": [
                    self.breathing_problem
                ],

                "Do you experience restless legs or involuntary movements during sleep?": [
                    self.restless_legs
                ],

                "Do you have any medical conditions that might affect your sleep?": [
                    self.medical_condition
                ],

                "Do you experience any of the following side effects from late sleeping?": [
                    side_effects
                ],

                "How often do you find it hard to concentrate due to lack of sleep?": [
                    self.concentration_problem
                ],

                "What strategies do you use to cope with the side effects of late sleeping?": [
                    self.coping_strategy
                ],

                "How would you rate the comfort of your sleeping environment": [
                    self._to_float(self.sleep_environment)
                ]
            }

            dataframe = DataFrame(input_dict)

            print("\n========== CREATED RAW DATAFRAME ==========")
            print(dataframe)

            return dataframe

        except Exception as e:
            raise CustomException(e, sys)


# =========================
# CLASSIFIER
# =========================
class SleepClassifier:

    def __init__(self):

        try:

            artifact_dir = "artifact"

            folders = [

                f for f in os.listdir(artifact_dir)

                if os.path.isdir(
                    os.path.join(artifact_dir, f)
                )
            ]

            if not folders:

                raise Exception(
                    "No trained model folder found"
                )

            latest_folder = sorted(folders)[-1]

            self.model_path = os.path.join(

                artifact_dir,
                latest_folder,
                "model_trainer",
                "trained_model",
                "model.pkl"
            )

        except Exception as e:
            raise CustomException(e, sys)

    # =========================
    # PREDICT
    # =========================
    def predict(self, dataframe):

        try:

            logging.info("Loading model...")

            if not os.path.exists(self.model_path):

                raise Exception(
                    f"Model not found at {self.model_path}"
                )

            with open(self.model_path, "rb") as f:

                model = pickle.load(f)

            logging.info(
                "Model loaded successfully"
            )

            print("\n========== MODEL LOADED ==========")

            result = model.predict(dataframe)

            print("\n========== MODEL PREDICTION ==========")
            print(result)

            return result

        except Exception as e:
            raise CustomException(e, sys)
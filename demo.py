from sleep_project.logger import logger

logger.info("Logging setup complete.")


from sleep_project.exception import CustomException
import sys

try:
    a = 10 / 0
except Exception as e:
    raise CustomException(e, sys)
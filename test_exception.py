from src.exception import CustomException
from src.logger import logging
import sys

try:
    logging.info("Starting exception test...")
    result = 10 / 0  # This will cause a ZeroDivisionError
except Exception as e:
    logging.error("An error occurred!", exc_info=True)
    raise CustomException(e, sys)

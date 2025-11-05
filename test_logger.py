from src.logger import logging, logger

print("Testing logger...")

logging.info("✅ Logging test using root logger")
logger.info("✅ Logging test using named logger")

print("If everything works, a log file should appear inside the 'logs' folder.")

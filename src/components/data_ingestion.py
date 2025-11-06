import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """Configuration class to store output file paths."""
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    """Handles reading, validating, and splitting data into train/test sets."""

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def initiate_data_ingestion(self, source_path: str = None):
      
        logging.info(" Starting Data Ingestion Process  ")
        try:
            # Step 1: Locate and read dataset
            if source_path is None:
                source_path = os.path.join(
                    os.getcwd(), "notebooks", "data", "Telco-Customer-Churn.csv"
                )

            logging.info(f"Reading dataset from: {source_path}")

            data = pd.read_csv(source_path)
            logging.info("Data successfully loaded into DataFrame")

            # Step 2: Basic data validation
            logging.info(f"Dataset shape: {data.shape}")
            logging.info(f"Columns: {list(data.columns)}")

            missing_values = data.isnull().sum().sum()
            if missing_values > 0:
                logging.warning(f" Dataset contains {missing_values} missing values")

            # Step 3: Create artifacts folder if not exists
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Step 4: Save raw data
            data.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Raw data saved at: {self.config.raw_data_path}")

            # Step 5: Train-test split
            logging.info("Initiating train-test split...")
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)
            logging.info(" Train and test data saved successfully")

            logging.info(" Data Ingestion Completed Successfully  ")

            return {
                "train_data_path": self.config.train_data_path,
                "test_data_path": self.config.test_data_path,
                "raw_data_path": self.config.raw_data_path
            }

        except Exception as e:
            logging.error("Error occurred during data ingestion", exc_info=True)
            raise CustomException(e, sys)

# let's test wether it's working or not
if __name__ == "__main__":
    try:
        logging.info(" Running Data Ingestion Pipeline...")

        # Initialize DataIngestion and run process
        ingestion = DataIngestion()
        paths = ingestion.initiate_data_ingestion()

        logging.info(" Data Ingestion Completed Successfully")
        print("Data Ingestion Completed. Files saved at:")
        print(f"Raw Data: {paths['raw_data_path']}")
        print(f"Train Data: {paths['train_data_path']}")
        print(f"Test Data: {paths['test_data_path']}")

    except Exception as e:
        logging.error(f"Pipeline failed due to: {str(e)}", exc_info=True)
        print(f"Pipeline failed due to: {str(e)}")

if __name__ == "__main__":
    try:
        logging.info(" Starting Data Ingestion Manually...")

        ingestion = DataIngestion()
        result_paths = ingestion.initiate_data_ingestion()

        print("\nData Ingestion Completed Successfully!")
        print("Files saved at:")
        print(f" Raw Data:   {result_paths['raw_data_path']}")
        print(f" Train Data: {result_paths['train_data_path']}")
        print(f" Test Data:  {result_paths['test_data_path']}")

        logging.info(" Data Ingestion Completed Successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"Pipeline failed: {str(e)}")


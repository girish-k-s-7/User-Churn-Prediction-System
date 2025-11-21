import sys
import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw_data.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")

        try:
            # Load your dataset
            data = pd.read_csv(
                os.path.join("Data Analysis", "data", "Telco_churn_after_DC.csv")
            )
            logging.info("Dataset Loaded Successfully")

            # Ensure artifacts folder exists
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            # Save raw data
            data.to_csv(self.raw_data_path, index=False)
            logging.info(f"Raw Data saved at: {self.raw_data_path}")

            # Train-test split
            train_set, test_set = train_test_split(
                data, test_size=0.2, random_state=42
            )

            # Save train and test files
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info("Train and Test data saved successfully")
            logging.info("Data Ingestion Completed")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logging.info("Error occurred during Data Ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("Data Ingestion Completed:")
    print("Train file saved at:", train_path)
    print("Test file saved at:", test_path)

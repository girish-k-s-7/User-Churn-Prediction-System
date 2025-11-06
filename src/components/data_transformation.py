import os
import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Starting the Data Transformation process")
        try:
            # Step 1: Load the data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Train and Test data loaded successfully")

            target_col = 'Churn'   # <-- change if your target column name differs

            # Step 2: Separate features and target
            X_train = train_data.drop(columns=[target_col])
            y_train = train_data[target_col].copy()
            X_test = test_data.drop(columns=[target_col])
            y_test = test_data[target_col].copy()

            # Step 3: Identify categorical and numerical columns
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

            logging.info(f"Categorical Columns: {cat_cols}")
            logging.info(f"Numerical Columns: {num_cols}")

            # Step 4: Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_cols),
                    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
                ],
                remainder='drop'
            )

            # Step 5: Fit and transform only X (not y)
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save preprocessor
            save_object(self.preprocessor_path, preprocessor)
            logging.info("Preprocessor saved at %s", self.preprocessor_path)

            logging.info("Data Transformation completed successfully")
            return X_train_transformed, X_test_transformed, y_train.values, y_test.values, self.preprocessor_path

        except Exception as e:
            logging.error("Error occurred during data transformation", exc_info=True)
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    paths = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = transformer.initiate_data_transformation(
        paths['train_data_path'], paths['test_data_path']
    )

    print("Data Transformation Completed!")
    print("Preprocessor saved at:", preprocessor_path)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

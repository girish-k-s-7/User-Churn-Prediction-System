import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        logging.info("Model Training Started")

        try:
            # Convert to numpy arrays
            X_train, X_test = np.array(X_train), np.array(X_test)
            y_train, y_test = np.array(y_train), np.array(y_test)

            logging.info(f"Data shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

            # Define candidate models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }

            # Evaluate all models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
            logging.info(f"Model Report: {model_report}")

            # Select best model based on F1-score
            best_model_name = max(model_report, key=lambda x: model_report[x]['f1_score'])
            best_model = models[best_model_name]
            best_metrics = model_report[best_model_name]

            logging.info(f"Best Model: {best_model_name} | F1-Score: {best_metrics['f1_score']:.4f}")

            # Retrain best model on full data
            best_model.fit(X_train, y_train)

            # Save model
            save_object(self.config.trained_model_path, best_model)
            logging.info(f"Model saved at: {self.config.trained_model_path}")

            return best_model_name, best_metrics

        except Exception as e:
            logging.error("Error during model training", exc_info=True)
            raise CustomException(e, sys)


# Test end-to-end pipeline
if __name__ == "__main__":
    ingestion = DataIngestion()
    paths = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = transformer.initiate_data_transformation(
        paths['train_data_path'], paths['test_data_path']
    )

    trainer = ModelTrainer()
    best_model, metrics = trainer.initiate_model_training(X_train, X_test, y_train, y_test)

    print("\n Training Completed!")
    print(f"Best Model: {best_model}")
    print("Metrics:", metrics)

import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
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
            X_train, X_test = np.array(X_train), np.array(X_test)
            y_train, y_test = np.array(y_train), np.array(y_test)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }

            # Step 1: Evaluate all models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
            best_model_name = max(model_report, key=lambda x: model_report[x]['f1_score'])
            best_model = models[best_model_name]

            logging.info(f"Base Best Model: {best_model_name} | F1: {model_report[best_model_name]['f1_score']:.4f}")

            # Step 2: Define hyperparameter grids
            param_grids = {
                "Logistic Regression": {
                    "C": [0.1, 0.5, 1, 5, 10],
                    "solver": ['liblinear', 'lbfgs']
                },
                "Decision Tree": {
                    "max_depth": [3, 5, 7, 10, None],
                    "min_samples_split": [2, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            }

            # Step 3: Apply GridSearchCV on the best model
            grid = GridSearchCV(
                estimator=best_model,
                param_grid=param_grids[best_model_name],
                cv=3,
                scoring='f1',
                verbose=1,
                n_jobs=-1
            )
            grid.fit(X_train, y_train)

            logging.info(f"Best Params for {best_model_name}: {grid.best_params_}")

            # Step 4: Evaluate tuned model
            tuned_model = grid.best_estimator_
            y_pred = tuned_model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, pos_label='Yes'),
                "recall": recall_score(y_test, y_pred, pos_label='Yes'),
                "f1_score": f1_score(y_test, y_pred, pos_label='Yes')
            }

            # Step 5: Save tuned model
            save_object(self.config.trained_model_path, tuned_model)
            logging.info(f"Tuned model saved at: {self.config.trained_model_path}")

            logging.info(f"Final Tuned Model: {best_model_name} | F1: {metrics['f1_score']:.4f}")
            return best_model_name, metrics

        except Exception as e:
            logging.error("Error during model training", exc_info=True)
            raise CustomException(e, sys)


# Test block
if __name__ == "__main__":
    ingestion = DataIngestion()
    paths = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = transformer.initiate_data_transformation(
        paths['train_data_path'], paths['test_data_path']
    )

    trainer = ModelTrainer()
    best_model, metrics = trainer.initiate_model_training(X_train, X_test, y_train, y_test)

    print("\n Hyperparameter Tuning Completed!")
    print(f"Best Model: {best_model}")
    print("Tuned Metrics:", metrics)

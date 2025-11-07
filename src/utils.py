import os
import sys
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
     
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f" Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
  
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Evaluate multiple models and return performance report.
    Calculates accuracy, precision, recall, and F1-score for each model.
    """
    report = {}
    try:
        for name, model in models.items():
            logging.info(f" Training and evaluating model: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            
            unique_labels = list(set(y_test))

            if "Yes" in unique_labels:
                pos_label = "Yes"
            elif 1 in unique_labels:
                pos_label = 1
            elif True in unique_labels:
                pos_label = True
            else:
                
                pos_label = unique_labels[-1]

        
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0)
                if len(unique_labels) == 2 else precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0)
                if len(unique_labels) == 2 else recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0)
                if len(unique_labels) == 2 else f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            report[name] = metrics
            logging.info(f" {name} → F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")

        return report

    except Exception as e:
        logging.error(" Error during model evaluation", exc_info=True)
        raise CustomException(e, sys)

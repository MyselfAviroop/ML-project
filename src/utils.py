# src/utils.py
import os
import sys
import pickle
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models with GridSearchCV.
    Handles classification and regression separately.
    """
    try:
        report = {}

        for name, model in models.items():
            # Determine if classification or regression
            is_classifier = hasattr(model, "predict_proba")

            # Determine proper CV
            if is_classifier:
                unique, counts = np.unique(y_train, return_counts=True)
                min_class_count = min(counts)
                cv_splits = max(2, min(3, min_class_count))
                cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            else:
                cv_splits = min(3, X_train.shape[0])
                cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

            # Get parameter grid
            para = param.get(name, {})  # Default empty dict if not provided

            # Run GridSearchCV
            gs = GridSearchCV(model, param_grid=para, cv=cv, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Set best params and refit
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_test_pred = model.predict(X_test)
            if is_classifier:
                score = accuracy_score(y_test, y_test_pred)
            else:
                score = r2_score(y_test, y_test_pred)

            report[name] = score

        return report

    except Exception as e:
        raise CustomException(e, sys)

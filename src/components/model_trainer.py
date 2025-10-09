from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from src.utils import save_object, evaluate_models
from src.logger import logger
from src.exception import CustomException
import os, sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameter grids
            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 500],
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.6, 0.8, 1.0],
                    "min_samples_split": [2, 5, 10]
                },
                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "normalize": [True, False]
                },
                "XGBRegressor": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "gamma": [0, 0.1, 0.2]
                },
                "CatBoosting Regressor": {
                    "iterations": [100, 200, 500],
                    "depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "l2_leaf_reg": [1, 3, 5, 7]
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "loss": ["linear", "square", "exponential"]
                }
            }

            # Evaluate all models with hyperparameter tuning
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Get the best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with score >= 0.6")

            logger.info(f"Best model: {best_model_name} with score: {best_model_score}")
            best_model.fit(X_train, y_train)

            # Save the trained model
            save_object(self.config.trained_model_file_path, best_model)

            # Evaluate on test data
            y_pred = best_model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            logger.info(f"R2 score on test data: {r2_square}")

            return r2_square

        except Exception as e:
            logger.error("Error occurred in model training stage")
            raise CustomException(e, sys)

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

    # <-- This method must be indented inside the class
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                # your parameter grid for GridSearchCV
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logger.info(f"Best model: {best_model_name} with score: {best_model_score}")
            best_model.fit(X_train, y_train)
            save_object(self.config.trained_model_file_path, best_model)

            y_pred = best_model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            logger.info(f"R2 score on test data: {r2_square}")

            return r2_square

        except Exception as e:
            logger.error("Error occurred in model training stage")
            raise CustomException(e, sys)

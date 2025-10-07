import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    target_column: str = 'math_score'
    numerical_columns: list = ('writing_score', 'reading_score')
    categorical_columns: list = (
        'gender', 'race_ethnicity', 'parental_level_of_education', 
        'lunch', 'test_preparation_course'
    )


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, self.config.numerical_columns),
                ('cat_pipeline', cat_pipeline, self.config.categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Read train and test data completed")
            logger.info(f"Train head:\n{train_df.head().to_string()}")
            logger.info(f"Test head:\n{test_df.head().to_string()}")

            preprocessing_obj = self.get_data_transformer_object()

            target_col = self.config.target_column
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            logger.info("Applying preprocessing on training and testing data")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            abs_path = os.path.abspath(self.config.preprocessor_obj_file_path)
            logger.info(f"Saving preprocessor object at: {abs_path}")
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessing_obj)
            logger.info(f"Preprocessor saved successfully at {abs_path}")

            return train_arr, test_arr, abs_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(ROOT_DIR, "artifacts", "train.csv")
    test_path = os.path.join(ROOT_DIR, "artifacts", "test.csv")

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
    print("Transformed train array shape:", train_arr.shape)
    print("Transformed test array shape:", test_arr.shape)
    print("Preprocessor saved at:", preprocessor_path)

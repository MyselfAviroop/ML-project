import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logger
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, source_data_path: str):
        logger.info("Entered the Data Ingestion method")
        try:
            if not os.path.exists(source_data_path):
                raise FileNotFoundError(f"Dataset not found at {source_data_path}")

            df = pd.read_csv(source_data_path)
            logger.info(f"Dataset loaded successfully with shape {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Raw dataset saved at {self.ingestion_config.raw_data_path}")

            logger.info("Performing train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info(f"Train dataset saved at {self.ingestion_config.train_data_path}")
            logger.info(f"Test dataset saved at {self.ingestion_config.test_data_path}")
            logger.info("Data Ingestion completed successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logger.error("Error occurred in Data Ingestion stage")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Step 0: Define source CSV
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        source_path = os.path.join(ROOT_DIR, "notebook", "data", "stud.csv")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion(source_path)
        print("Train CSV Path:", train_path)
        print("Test CSV Path:", test_path)

        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)
        print("Transformed train array shape:", train_arr.shape)
        print("Transformed test array shape:", test_arr.shape)
        print("Preprocessor saved at:", preprocessor_path)

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        r2_score_value = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print("R2 score on test data:", r2_score_value)

    except Exception as e:
        print("Pipeline failed:", e)

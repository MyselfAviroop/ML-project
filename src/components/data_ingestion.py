import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logger

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, source_data_path: str):
        logger.info("Entered the data ingestion method")
        try:
            if not os.path.exists(source_data_path):
                raise FileNotFoundError(f"Dataset not found at {source_data_path}")

            df = pd.read_csv(source_data_path)
            logger.info(f"Dataset loaded successfully with shape {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info("Raw dataset saved")

            logger.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_path = os.path.join(ROOT_DIR, "notebook", "data", "stud.csv")

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion(source_path)
    print("Train Data Path:", train_data)
    print("Test Data Path:", test_data)

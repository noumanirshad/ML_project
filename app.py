from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException
import sys
from src.ML_Project.components.data_ingestion import DataIngestion
from src.ML_Project.components.data_transformation import DataTransformation
from src.ML_Project.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    logging.info("The exsecution project has been created")


    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        
        # print(f"{X_train[:5,:]}, \n{X_test.shape}")
        model_trainer = ModelTrainer()
        model_trainer.trainer_initiate(X_train, X_test, y_train, y_test)



    except Exception as e:
        logging.info(f"An exception has occurred : {e}")
        raise CustomException(e, sys)
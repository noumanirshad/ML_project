from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException
import sys
from src.ML_Project.components.data_ingestion import DataIngestion


if __name__ == '__main__':
    logging.info("The exsecution project has been created")


    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
        

    except Exception as e:
        logging.info(f"An exception has occurred : {e}")
        raise CustomException(e, sys)
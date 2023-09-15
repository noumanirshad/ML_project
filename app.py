from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException
import sys


if __name__ == '__main__':
    logging.info("The exsecution project has been created")


    try:
        a = 1/0

    except Exception as e:
        logging.info(f"An exception has occurred : {e}")
        raise CustomException(e, sys)
from src.banglorepriceprediction.logger import logging
from src.banglorepriceprediction.exception import CustomException
import sys



if __name__=="__main__":
    logging.info("Execution has started")



    try:
        a=1/0
    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)        
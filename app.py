from src.banglorepriceprediction.logger import logging
from src.banglorepriceprediction.exception import CustomException
from src.banglorepriceprediction.components.data_ingestion import DataIngestion
import sys



if __name__=="__main__":
    logging.info("Execution has started")



    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()

        



    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)        
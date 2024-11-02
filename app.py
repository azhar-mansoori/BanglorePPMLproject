from src.banglorepriceprediction.logger import logging
from src.banglorepriceprediction.exception import CustomException
from src.banglorepriceprediction.components.data_ingestion import DataIngestion
from src.banglorepriceprediction.components.data_transformation import DataTransformation
import sys



if __name__=="__main__":
    logging.info("Execution has started")



    try:
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        



    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)        
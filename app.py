from src.banglorepriceprediction.logger import logging
from src.banglorepriceprediction.exception import CustomException
from src.banglorepriceprediction.components.data_ingestion import DataIngestion
from src.banglorepriceprediction.components.data_transformation import DataTransformation
from src.banglorepriceprediction.components.model_tranier import ModelTrainer
import sys



if __name__=="__main__":
    logging.info("Execution has started")



    try:
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation=DataTransformation()
        train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

        



    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)        
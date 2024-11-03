import sys
import os
import pandas as pd
from src.banglorepriceprediction.exception import CustomException
from src.banglorepriceprediction.logger import logging
from src.banglorepriceprediction.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            print("before loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("after loading")

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,location:str,
                      total_sqft:str,
                      bhk:str,
                      bath:float):
        self.location=location
        self.total_sqft=total_sqft
        self.bhk=bhk
        self.bath=bath

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "location":[self.location],
                "total_sqft":[self.total_sqft],
                "bhk":[self.bhk],
                "bath":[self.bath],
            } 

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
                    
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
from src.banglorepriceprediction.exception import CustomException
from src.banglorepriceprediction.logger import logging
from src.banglorepriceprediction.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
           

            numerical_columns=["total_sqft","bath","bhk"]
            categorical_columns=["location"]


            #handaling missing values & doing scaling

            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ])
            logging.info(f"categorical columns:{categorical_columns}")
            logging.info(f"numerical columns:{numerical_columns}")

            #combining both the pipeline 

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("reading the test and train file")

            preprocessor_obj=self.get_data_transformer_object()      #object create / initialize

            target_column_name="price"
            numerical_columns=["total_sqft","bath","bhk"] 

            #dividing train dataset into independent and dependent feture 
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)  #independent featue 
            target_feature_train_df=train_df[target_column_name]                       #dependent feaure i.e. price
            

            #now for test dataset
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)  #independent featue 
            target_feature_test_df=test_df[target_column_name]                       #dependent feaure i.e. price

            logging.info("applying preprocessing on training & test dataframe")

            #aplying transformation
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)


            #combinig both input and target to get complete train & test dataset

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]            
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessor_obj
            )

            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(sys,e)
        







   

























        except Exception as e:
            raise  CustomException(e,sys)   
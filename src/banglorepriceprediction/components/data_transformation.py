'''import sys
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
                #("imputer",SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                #("imputer",SimpleImputer(strategy="most_frequent")),
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
            target_feature_train_df=train_df[target_column_name]   
            
            print(input_feature_train_df) 
            print(target_feature_train_df)                   #dependent feaure i.e. price
           

            #now for test dataset
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)  #independent featue 
            target_feature_test_df=test_df[target_column_name]                       #dependent feaure i.e. price

            logging.info("applying preprocessing on training & test dataframe")

            #aplying transformation
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            
            logging.info("preprocessing complete ")

            
            print(np.array(target_feature_train_df))
            print(input_feature_train_arr)




            
            #train_target_arr=np.array(target_feature_train_df)
            #test_target_arr=np.array(target_feature_test_df)

           
            #combinig both input and target to get complete train & test dataset

            #train_arr=np.c_[input_feature_train_arr,train_target_arr]            
            #test_arr=np.c_[input_feature_test_arr,test_target_arr]
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessor_obj
            )

            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(sys,e)'''









import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.banglorepriceprediction.utils import save_object

from src.banglorepriceprediction.exception import CustomException
from src.banglorepriceprediction.logger import logging
import os



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            numerical_columns = ["total_sqft", "bath","bhk"]
            #categorical_columns = [""]
                
            
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())

            ])
            cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("level_encoder",LabelEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ])

            #logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                   # ("cat_pipeline",cat_pipeline,categorical_columns)
                ]

            )
            return preprocessor
            

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="price"
            numerical_columns = ["total_sqft", "bath","bhk"]

            ## divide the train dataset to independent and dependent feature

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            print(np.array(target_feature_train_df))
            print(np.array(input_features_train_df))

            


            ## divide the test dataset to independent and dependent feature

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            print(np.array(target_feature_train_df))
            print(input_feature_train_arr)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )








        except Exception as e:
            raise CustomException(sys,e)

        







   

























      
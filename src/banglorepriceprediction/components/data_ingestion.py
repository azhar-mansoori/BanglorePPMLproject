import os
import sys
from src.banglorepriceprediction.exception import CustomException
from src.banglorepriceprediction.logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.banglorepriceprediction.utils import convertRange,remove_outliers_sqft,bhk_outliers_remover
#from src.banglorepriceprediction.components.data_transformation import DataTransformation




@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df=pd.read_csv(os.path.join('notebook/data','Bengaluru_House_Data.csv'))
            df=df.drop(['area_type','availability','balcony','society'],axis=1)

            logging.info("checking null vales ")
            #obj=DataTransformation
            #obj1=obj.get_data_transformer_object(self)
            print(df)
            #target_column_name="price"
            #input_feature_train_df=df.drop(columns=[target_column_name],axis=1)



            #df=obj1.fit_transform(input_feature_train_df)
            df=df.dropna()

            print(df[df['size'].isnull()])

            logging.info("null values found:{df[df['size'].isnull()]}")

             
            df['bhk']=df['size'].str[0:2]
            df['bhk']=df['bhk'].astype(str).astype(int)          
            logging.info("size change to int")


            df['total_sqft'] = df['total_sqft'].apply(convertRange)
            logging.info("total sqrt transforms:")


            #df['location'] = df['location'].apply(location_remove)
            #df['location'] = df['location'].apply(lambda x: x.strip())
            df['location'] = df['location'].astype(str).str.replace('\D+', '')
            location_stats = df['location'].value_counts(ascending=False)
            location_stats_less_than_10 = location_stats[location_stats<=10]
            df.location = df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)


            logging.info("location reduced")
           #print("type is:",df['total_sqft'])

            df = df[((df['total_sqft']/df['bhk'])>=300)]
            logging.info("reduce the data when room less than 300sqrt")

            df['price_per_sqrt'] = df['price']*100000/df['total_sqft']
            logging.info("price per sqrt created ")

            df = remove_outliers_sqft(df)
            logging.info("outliers remove w.r.t. price per sqrt")

            df = bhk_outliers_remover(df)
            logging.info("bhk outliers remove")

            df=df.drop(['size','price_per_sqrt'],axis=1)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True ) 

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)  
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("data ingestion is complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )  
        except Exception as e:
            raise CustomException(e,sys)



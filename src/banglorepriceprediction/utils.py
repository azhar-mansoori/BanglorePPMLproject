import os
import sys
from src.banglorepriceprediction.exception import CustomException
from src.banglorepriceprediction.logger import logging
import pandas as pd
import numpy as np
import pickle




def convertRange(x):

    temp= x.split('-')
    if len(temp) == 2:
        return (float(temp[0])+float(temp[1]))/2
    try:
        return float(x)
    except:
        return None
    
'''
def location_remove(x):
    temp=x.strip()
    temp_count=len(temp)
    temp_less_10 = temp_count[temp_count<=10]
    if x in temp_less_10:
        x = 'others'
    else:
        return x 
'''


def remove_outliers_sqft(df):
    df_output= pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqrt)
        st = np.std(subdf.price_per_sqrt)

        gen_df =subdf[(subdf.price_per_sqrt>(m-st)) & (subdf.price_per_sqrt<=(m+st))]
        df_output=pd.concat([df_output,gen_df],ignore_index=True)
        return df_output




def bhk_outliers_remover(df):
    exlude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqrt),
                'std':np.std(bhk_df.price_per_sqrt),
                'count':bhk_df.shape[0]

            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exlude_indices=np.append(exlude_indices,bhk_df[bhk_df.price_per_sqrt<(stats['mean'])].index.values)
    return df.drop(exlude_indices,axis='index')            



def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)        
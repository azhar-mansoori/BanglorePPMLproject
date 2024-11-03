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


#MAKING A FLASK APPLICATION TO GET NEW DATA AND PREDICT THE VALUE

from flask import Flask,render_template,request,Request
from src.banglorepriceprediction.pipelines.prediction_pipeline import PredictPipeline,CustomData
from sklearn.preprocessing import StandardScaler
import json

application= Flask(__name__)

app = application
__locations = None
__data_columns = None

f = open('columns.json')
__data_columns = json.loads(f.read())['data_columns']
__locations = __data_columns[3:]
#print(__locations)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
    if request.method=='GET':
        return render_template('home.html' ,locations=__locations)
    else:
        data=CustomData(
           
            location=request.form.get('sLocation'),
            total_sqft=request.form.get('Squareft'),
            bhk=request.form.get('uiBHK'),
            bath=float(request.form.get('uiBathrooms'))
        )                         

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("before prediction")

        predict_pipeline=PredictPipeline()
        print("mid prediction")

        results = predict_pipeline.predict(pred_df)
        print("after prediction")
        print("the result is :",results)

        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")        
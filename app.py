from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mangum import Mangum
import uvicorn
import pandas as pd
import pickle
import boto3
from flaml import *

class PredictPipeline:
    def __init__(self):
        self.cities = {''}
        self.s3 = boto3.resource(service_name = 's3', 
                            region_name = 'us-east-2',
                            aws_access_key_id = 'AKIA4BCAJ6NVR7R4S6VO',
                            aws_secret_access_key = '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')
        self.bucket = 'climformatics'

    def get_data(self, city, start_time, end_time):
        obj = self.s3.Bucket(self.bucket).Object(f'{city}_train.csv').get()
        df = pd.read_csv(obj['Body'])
        df.time = pd.to_datetime(df.time)
        if start_time == end_time:
            return df[df.time == start_time]
        else:
            return df[(df.time >= start_time) & (df.time <= end_time)]
        
    def predict(self, city, start_time, end_time):
        data = self.get_data( city, start_time, end_time)
        data = data.drop(columns = ['FWI','time'])
        # print("Before Loading")

        response = self.s3.Bucket(self.bucket).Object(f'_{city}_model.pkl').get()
        body = response['Body'].read()
        model = pickle.loads(body)

        # print("After Loading")
        preds=model.predict(data)
        return preds
        


app = FastAPI()
handler = Mangum(app)

@app.get('/predict')
async def predict(city:str, start_time:str, end_time:str):

    preds = PredictPipeline().predict(city, start_time, end_time)#.values
    return JSONResponse({'fwi':preds.tolist()})

# if __name__ == '__main__':
#     uvicorn.run(app)
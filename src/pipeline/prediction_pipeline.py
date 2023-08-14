import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
import pickle
import boto3

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
        try:
            data = self.get_data( city, start_time, end_time)
            data = data.drop(columns = ['FWI','time'])
            # print("Before Loading")

            response = self.s3.Bucket(self.bucket).Object(f'_{city}_model.pkl').get()
            body = response['Body'].read()
            model = pickle.loads(body)

            # print("After Loading")
            preds=model.predict(data)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
import os
import sys
from pathlib import Path

# As PosixPath
sys.path.append(os.path.realpath('.'))
# print(sys.path)
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


def transform_forecast_data(df):
        if 'Unnamed: 0' in df.columns:
            df.drop(columns='Unnamed: 0', inplace=True)
        df['time'] = pd.to_datetime(df['time'])
        df = df.rename(columns = {col:'fut_'+col for col in df.columns if col != 'time'})
        df = df.rename(columns = {'fut_tmp':'fut_temp' })
        df['fut_temp'] = (df['fut_temp'] - 273.15) * (9/5) + 32
        df['fut_uwind'] = df['fut_uwind'] * 2.23694
        df['fut_vwind'] = df['fut_vwind'] * 2.23694
        df['fut_wind'] = np.sqrt((df['fut_uwind'] ** 2) + (df['fut_vwind'] ** 2))
        return df
    
def transform_historical_data(df):
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns = {'Date':'time',
                        'Air Temp (F)':'hist_temp',
                        'Rel Hum (%)':'hist_rh',
                        'Wind Speed (mph)':'hist_ws'},
                        inplace = True)
    shifted_dfs = [df]
    for i in range(-4,5):
        offset = df.shift(24 * (365 + i)).drop(columns = 'time')
        offset.columns = [col + str(i) for col in offset.columns]
        shifted_dfs.append(offset)

    df = pd.concat(shifted_dfs, axis = 1)

    df['hour'] = np.sin(df.time.dt.hour / 24)
    df['dow'] = np.sin(df.time.dt.day_of_week / 7)
    df['doy'] = np.sin(df.time.dt.day_of_year / 365)
    df['month'] = np.sin(df.time.dt.month / 12)
    
    return df


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")
    raw_data_path: str=os.path.join('artifact',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.cities = ['Mariposa', 'Ramona','Redding',
                          'Sanel','Santa-Barbara','Union-City']
        self.city_files = {city:[] for city in self.cities}

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion method or component")
        try:
            files = os.listdir('data/hourly')
            for data_file in files:
                name_split = data_file.split('_')[0]
                self.city_files[name_split].append(os.path.join('data/hourly',data_file)) 

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path,), exist_ok=True)
            
            for city in self.city_files:
                    file1, file2 = self.city_files[city]
                    df1 = pd.read_csv(file1, ) if '6' in file1 else pd.read_csv(file2)
                    df2 = pd.read_csv(file2) if '6' in file1 else pd.read_csv(file1)

                    df1 = transform_forecast_data(df1)
                    df2 = transform_historical_data(df2)
                    print(df1)
                    print(df2)
                    raw_df = pd.merge(df1, df2, how = 'inner', on = 'time')
                    raw_df = raw_df.dropna()
                    raw_df.to_csv(f"artifact/{city}_raw.csv", index = False, header = True)
                                
                    logging.info(f"Train test split for {city} initiated")
                    train_set, test_set = train_test_split(raw_df, test_size=0.2, random_state=42)

                    train_set.to_csv(f"artifact/{city}_train.csv", index = False)
                    test_set.to_csv(f"artifact/{city}_test.csv", index = False)
            logging.info("Data Ingestion ccompleted")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)


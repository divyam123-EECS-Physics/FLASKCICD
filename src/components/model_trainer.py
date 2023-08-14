import os
import sys
from dataclasses import dataclass
sys.path.append(os.path.realpath('.'))

import pandas as pd

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost  import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from flaml import AutoML
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path  = os.path.join("artifact", "_{}_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.cities = ['Mariposa', 'Ramona','Redding',
                          'Sanel','Santa-Barbara','Union-City']

    def initiate_model_trainer(self):
        try:
            scores = []
            for city in self.cities:
                
                logging.info(f"{city}:splitting training and val input data")
                
                train_df = pd.read_csv(f"artifact/{city}_train.csv")
                X_train = train_df.drop(columns = ['time','FWI']).values
                y_train = train_df['FWI'].values
                
                test_df = pd.read_csv(f"artifact/{city}_test.csv")
                X_test = test_df.drop(columns = ['time','FWI']).values
                y_test = test_df['FWI'].values
                automl = AutoML()
                # Specify automl goal and constraint
                automl_settings = {
                    "time_budget": 120,  # in seconds
                    "metric": 'mae',
                    "task": 'regression',
                }
                # Train with labeled input data
                automl.fit(X_train=X_train, y_train=y_train,
                        **automl_settings)
                
                # models = {
                #     "Random Forest": RandomForestRegressor(),
                #     "Decision Tree": DecisionTreeRegressor(),
                #     "Gradient Boosting": GradientBoostingRegressor(),
                #     "Linear Regression": LinearRegression(),
                #     "XGBRegressor": XGBRegressor(),
                #     "AdaBoost Regressor": AdaBoostRegressor(),
                # }

                # params={
                #     "Decision Tree": {
                #         'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                #         # 'splitter':['best','random'],
                #         # 'max_features':['sqrt','log2'],
                #     },
                #     "Random Forest":{
                #         # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    
                #         # 'max_features':['sqrt','log2',None],
                #         'n_estimators': [8,16,32,64,128,256]
                #     },
                #     "Gradient Boosting":{
                #         # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                #         'learning_rate':[.1,.01,.05,.001],
                #         'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                #         # 'criterion':['squared_error', 'friedman_mse'],
                #         # 'max_features':['auto','sqrt','log2'],
                #         'n_estimators': [8,16,32,64,128,256]
                #     },
                #     "Linear Regression":{},
                #     "XGBRegressor":{
                #         'learning_rate':[.1,.01,.05,.001],
                #         'n_estimators': [8,16,32,64,128,256]
                #     },
                #     "AdaBoost Regressor":{
                #         'learning_rate':[.1,.01,0.5,.001],
                #         # 'loss':['linear','square','exponential'],
                #         'n_estimators': [8,16,32,64,128,256]
                #     }
                    
                # }

                # logging.info(f"{city}:Evaluating models")

                # model_report:dict = evaluate_models(X_train = X_train, 
                #                                 y_train = y_train, 
                #                                 X_test = X_test, 
                #                                 y_test = y_test, 
                #                                 models = models,
                #                                 params = params)
                # best_score = 0
                # best_model_name = ''
                # # getting best model from dictionary
                # for model in model_report:
                #     score = model_report[model]
                #     if score > best_score:
                #         best_model_name = model
                #         best_score = score


                best_model = automl.model#models[best_model_name]
                # if best_score < 0.6:
                #     raise CustomException("no best model found", sys)
                # logging.info(f"best model found {best_model_name}")

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path.format(city), 
                    obj=best_model
                )

                predicted = best_model.predict(X_test)
                score = r2_score(y_test, predicted)
                scores.append(score)
            return score
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    di = ModelTrainer()
    di.initiate_model_trainer()
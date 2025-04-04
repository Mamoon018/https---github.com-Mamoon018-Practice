
import os 
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (

    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from scikeras.wrappers import KerasRegressor

from src.exception import customException
from src.logger import logging
from src.utils import save_object, evaluate_models
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trainer_model_configfile = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiaing_model_trainer(self,train_array,test_array, rul):
        try: 
            logging.info('Splitting training & test input data')

            X_train, Y_train, X_test, Y_test = (

                train_array[:,:-1], # All columns of train_array except last column
                train_array[:,-1], # only last column extracted
                test_array,
                rul
                

            ) 
            
            logging.info('specifying models')
            ML_models = {
                
                'CNN-LSTM' : 'keras_model',
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGB Regressor': XGBRegressor(),
                'CatBoost Regressor': CatBoostRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor()

            }

            tuning_parameters = {

                
                "K-Neighbors Regressor": {
                    #'n_neighbors': [ 5,  10],
                    #'weights': ['uniform', 'distance'],  # Keep for regression
                    #'p': [1, 2]  # Manhattan (p=1) vs Euclidean (p=2) distance
                },
    
                "XGB Regressor": {
                    'n_estimators': [50, 100, 200],  # More trees
                    'learning_rate': [0.01, 0.05, 0.1],  # Wider range
                    'max_depth': [3, 5, 7],  # Control tree depth
                    'subsample': [0.7, 0.9, 1.0],  # Prevent overfitting
                    'colsample_bytree': [0.7, 0.9, 1.0]  # Feature sampling
                },

                "CatBoost Regressor": {
                    'iterations': [10],
                    #'learning_rate': [0.1],
                    #'depth': [6, 8, 10],
                    #'l2_leaf_reg': [1, 3, 5, 7, 9]  # Regularization for regression
                },

                "AdaBoost Regressor": {
                    'n_estimators': [8,],
                    #'learning_rate': [0.1],
                    #'loss': ['linear', 'square', 'exponential']  # Regression-specific
                },

                "Gradient Boosting Regressor": {
                    'n_estimators': [16],
                    #'learning_rate': [ 0.01],
                    #'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    #'max_depth': [3, 5, 7, 10],
                    #'min_samples_leaf': [1, 2, 4],
                    #'loss': ['squared_error', 'absolute_error']  # Regression losses
                },

                "Random Forest Regressor": {
                    'n_estimators': [8, 16],
                    #'max_depth': [None, 10, 20, 30, 40, 50],
                    #'min_samples_split': [2, 5, 10],
                    #'min_samples_leaf': [1, 2, 4],
                    #'max_features': ['sqrt', 'log2', None],
                    #'criterion': ['squared_error', 'absolute_error']  # Regression criteria
                },

                "Decision Tree Regressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],  # Regression criteria
                    #'splitter': ['best', 'random'],
                    #'max_depth': [None, 10, 20, 30, 40, 50],
                    #'min_samples_split': [2, 5, 10],
                    #'min_samples_leaf': [1, 2, 4],
                    #'max_features': ['sqrt', 'log2', None]
                }
            }


            model_report:dict = evaluate_models(x_train= X_train, y_train= Y_train , x_test= X_test, y_test=Y_test , models= ML_models, param= tuning_parameters)


             ## We will extract the best model score from report

            Best_model_score = max(sorted(model_report.values()))

            ## Best model name

            Best_model_name = list(model_report.keys())[ 
                list(model_report.values()).index(Best_model_score)
                                                        ]
            best_model = ML_models[Best_model_name] #instance

            #if Best_model_score < 0.6:
                #raise ValueError("Not a single model model performed well")
            logging.info(f"Best model {Best_model_name} with score {Best_model_score} found on training and test dataset")

            save_object(

                file_path = self.model_trainer_config.trainer_model_configfile,
                obj = best_model                 
            )


            y_predicted = best_model.predict(X_test)
            accuracy = r2_score(Y_test,y_predicted)
            return accuracy, Best_model_name, Best_model_score, best_model, model_report

        except Exception as e:
            raise customException(e,sys)

# Feature engineering 
# Outliers 
# imputation 
# feature selection 

import sys 
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.impute import KNNImputer
from src.exception import customException
from src.logger import logging
import os 
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import save_object

@dataclass
class datatransformationconfig:
    preprocessing_obj_file: str = os.path.join('artifacts', 'preprocessing.pkl')



class feature_engineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        
        self.column_name = ['ops_1', 'ops_2', 'ops_3'] + [f'sensor_{i}' for i in range(1,22)]
        self.stats_columns = ['sensor_11','sensor_12', 'sensor_2','sensor_4', 'sensor_7', 'sensor_8', 'sensor_9']
        self.delta_columns = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 
                 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 
                 'sensor_20', 'sensor_21']

    def fit(self,X, Y = None):

        return self
    
    def transform(self, X, y= None):

        try:
            logging.info("feature eng execution will start")
            X = X.copy()

        # Assigning the names to the columns 
            X.columns = self.column_name
        # Calculating RUL column
            logging.info("now, generating RUL column")
            calculated_cycle = X.groupby('unit_number')['times_in_cycles'].max().rename('Maximum_cycle')
            X = X.merge(calculated_cycle, on = 'unit_number')
            X['RUL'] = X['Maximum_cycle'] - X['times_in_cycles']
        # generating rolling stats 
            logging.info("now, generating rolling stats columns")
            for col in self.stats_columns:
                X[f"{col}_stats"] = X.groupby('unit_number')[col].transform(lambda x: x.rolling(window=10, min_periods=1).mean())

            logging.info("now, generating delta columns")
        # generating delta columns
            for col in self.delta_columns:
                X[f'{col}_delta'] = X.groupby('unit_number')[col].transform(lambda x: x.diff().fillna(0))

            return X 
    

        except Exception as e:
            raise customException(e,sys)
        
class sd_outlier_removal(BaseEstimator,TransformerMixin):

    def __init__(self, factor=3):
        self.factor = factor
        self.lower_limit = None
        self.upper_limit = None
        self.sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_11', 'sensor_7', 'sensor_8', 'sensor_14',  'sensor_15', 'sensor_20', 'sensor_21']
    
    def fit(self, X, y=None):
        
        selected_columns = X[self.sensors].values
        cal_mean = np.mean(selected_columns,axis=0)
        cal_std = np.std(selected_columns,axis=0)

        self.lower_limit = cal_mean - self.factor * cal_std
        self.upper_limit = cal_mean + self.factor * cal_std

        return self 
    
    def transform(self, X, y=None):
        try: 
            logging.info("executing sd outlier")
            X = X.copy()

            if self.lower_limit is None or self.upper_limit is None:
                raise ValueError('Outliers has not fitted yet')
            
            X_numeric = X[self.sensors]

            mask = (X_numeric >= self.lower_limit ) & (X_numeric <= self.upper_limit)

            X[self.sensors] = X_numeric.where(mask,other = np.nan)

            logging.info('Outliers removed using IQR method for selected columns')
            
            return X
        
        except Exception as e:
            raise customException(e,sys)

class IQR_outlier_removal(BaseEstimator, TransformerMixin):


    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_limit = None 
        self.upper_limit = None
        self.sensors = ['sensor_12', 'sensor_9', 'sensor_17']
    
    def fit(self, X, y=None):
        try: 
            logging.info("fitting IQR cal")
            X_selected_columns = X[self.sensors].values
            Q1 = np.percentile(X_selected_columns, 25, axis=0)
            Q3 = np.percentile(X_selected_columns,75, axis = 0)

            IQR = Q3 - Q1 
            self.lower_limit = Q1 - self.factor * IQR
            self.upper_limit = Q3 + self.factor * IQR
        
            return self
        except Exception as e:
            raise customException(e,sys)
    
    def transform(self,X, y=None):
        
        try:

            logging.info("IQR outlier remover will be executed")
            X = X.copy()
            X_selected_columns = X[self.sensors]

            mask = (X_selected_columns >= self.lower_limit) & (X_selected_columns <= self.upper_limit)

            # using .where we will replace values with NaN

            X[self.sensors] = X[X_selected_columns].where(mask, other = np.nan)

            logging.info("IQR Completed")

            return X 
            
        except Exception as e:
            raise customException(e,sys)


class imputing_Nan_using_KNN(BaseEstimator, TransformerMixin):


    def __init__(self, column_list, n_neighbors = 5):
            self.column_list = column_list
            self.n_neighbors = n_neighbors
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
            
    def fit(self,X, y = None):
            logging.info("getting strongly correlated columns for KNN models")
            df_corr = X.corr()
            highly_corr_columns = set()

            for col in self.column_list:
                for row in df_corr.index:
                    if row != col and abs(df_corr.loc[col,row]) >0.70:
                        highly_corr_columns.add(row)

            self.final_columns = list(set(self.column_list + list(highly_corr_columns)))
            self.imputer.fit(X[self.final_columns])  
            return self
        
    def transform(self, X):
        logging.info("initiating imputation after fitting executed")
        try:
            X = X.copy()
            df_subset = X[self.final_columns]
            df_imputed = self.imputer.transform(df_subset)
            X[self.final_columns] = df_imputed
            return X 
    
        except Exception as e:
            raise customException(e,sys)


class data_transformation:
    def __init__(self):
        self.config = datatransformationconfig()
    
    def get_datapreprocessing(self):
    
        try:
            
            feature_eng_steps = feature_engineering()
            numerical_columns = ['ops_1', 'ops_2', 'ops_3'] + [f'sensor_{i}' for i in range(1,22)]

            numerical_pipeline = Pipeline(steps=[
                
                ("std_outlier_removal",sd_outlier_removal()),
                ("IQR_outlier_removal", IQR_outlier_removal()),
                ("scalar", StandardScaler())
            ])

            preprocessor = Pipeline(steps=[
                            ("feature_engineering", feature_eng_steps),
                            ColumnTransformer(transformers=[("num_pipeline", numerical_pipeline, numerical_columns)
                            ])
                                    ])

            logging.info('preprocessor is ready to transform the data')

            return preprocessor
        except Exception as e:
            raise customException(e,sys)
    
    def transform_data(self, train_data, test_data):

        try:

            # read the data provided in the input 
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            # lets call the result of the previous method, preprocessor. It will give us columntransformer settings
            # that we will be able to apply on the datasets.

            preprocessor = self.get_datapreprocessing()

            # Now, before applying column transformer we need to get datasets into the form of the 
            # X_train, Y_train, X_test, Y_test

            target_column = 'RUL'
            X_train = train_df.drop(columns=[target_column], axis=1)
            Y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column], axis=1)
            Y_test = test_df[target_column]

            input_feature_train_array = preprocessor.fit_transform(X_train)
            input_feature_test_array = preprocessor.transform(X_test)

            logging.info('Preprocessor applied')

            # combining the transformed datasets and target variable

            train_arr = np.c_[input_feature_train_array,Y_train]
            test_arr = np.c_[input_feature_test_array, Y_test]

            # saving the preprocessor object 

            save_object(file_path=self.config.preprocessing_obj_file, obj = preprocessor)
            logging.info('saved preprocessor')

            return train_arr, test_arr, self.config.preprocessing_obj_file

        except Exception as e:
            logging.error('Error in data transformation')
            raise customException(e,sys)












    ######    UTILS      ######


    

import os 
import sys 

import numpy as np 
import pandas as pd 
import pickle

from src.exception import customException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise customException(e,sys)

def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):

            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            




            gs = GridSearchCV(model,para, cv = 3, n_jobs=-1)
            gs.fit(x_train,y_train) # train with hyperparameter tuning

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train) # train with hyperparameter tuning

            y_train_predict = model.predict(x_train)

            y_test_predict = model.predict(x_test)

            test_model_Score = r2_score(y_test, y_test_predict)

            report[list(models.keys())[i]] = test_model_Score


        return report
    
    except Exception as e:
        raise customException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise customException(e, sys)
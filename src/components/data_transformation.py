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
        
        self.column_name = ['unit_number', 'times_in_cycles', 'ops_1', 'ops_2', 'ops_3'] + [f'sensor_{i}' for i in range(1,22)]
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

            X.columns = self.column_name

            if not set(['unit_number', 'times_in_cycles']).issubset(X.columns):
                raise ValueError('Columns do not exist')
            
            logging.info("now, generating RUL column")
            calculated_cycle = X.groupby('unit_number')['times_in_cycles'].max().rename('Maximum_cycle')
            X = X.merge(calculated_cycle, on = 'unit_number')
            X['RUL'] = X['Maximum_cycle'] - X['times_in_cycles']

            logging.info("now, generating rolling stats columns")
            for col in self.stats_columns:
                if col in X.columns:
                    X[f"{col}_stats"] = X.groupby('unit_number')[col].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
                else:
                    raise ValueError(f'Column {col} does not exist')

            logging.info("now, generating delta columns")
            for col in self.delta_columns:
                if col in X.columns:
                    X[f'{col}_delta'] = X.groupby('unit_number')[col].transform(lambda x: x.diff().fillna(0))
                else:
                    raise ValueError(f'Column {col} does not exist')
            
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
        if not set(self.sensors).issubset(X.columns):
            raise ValueError('Columns do not exist')
        
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

            logging.info('Outliers removed using SD method for selected columns')
            
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
        if not set(self.sensors).issubset(X.columns):
            raise ValueError('Columns do not exist')
        
        logging.info("fitting IQR cal")
        X_selected_columns = X[self.sensors].values
        Q1 = np.percentile(X_selected_columns, 25, axis=0)
        Q3 = np.percentile(X_selected_columns,75, axis = 0)
        IQR = Q3 - Q1 
        self.lower_limit = Q1 - self.factor * IQR
        self.upper_limit = Q3 + self.factor * IQR
    
        return self
    
    def transform(self,X, y=None):
        try:
            logging.info("IQR outlier remover will be executed")
            X = X.copy()
            X_selected_columns = X[self.sensors]
            mask = (X_selected_columns >= self.lower_limit) & (X_selected_columns <= self.upper_limit)
            X[self.sensors] = X_selected_columns.where(mask, other = np.nan)
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
            
            numerical_columns = ['ops_1', 'ops_2', 'ops_3'] + [f'sensor_{i}' for i in range(1,22)]

            numerical_pipeline = Pipeline(steps=[
                
                ("std_outlier_removal",sd_outlier_removal()),
                ("IQR_outlier_removal", IQR_outlier_removal()),
                ("knn_imputation", imputing_Nan_using_KNN(column_list=numerical_columns)),
                ("scalar", StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[("num_pipeline", numerical_pipeline, numerical_columns)
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
            target_column = 'RUL'
            preprocessor = self.get_datapreprocessing()

            # Now, before applying column transformer we need to get datasets into the form of the 
            # X_train, Y_train, X_test, Y_test

            calculated_columns = feature_engineering()
            train_df = calculated_columns.fit_transform(train_df)
            test_df = calculated_columns.transform(test_df)

            X_train = train_df.drop(columns=target_column, axis=1)
            Y_train = train_df[target_column]
            
            # For test data, extract the last cycle for each engine
            test_df = test_df.groupby('unit_number').last().reset_index()
            X_test = test_df.drop(columns=target_column, axis=1)

            input_feature_train_array = preprocessor.fit_transform(X_train)
            input_feature_test_array = preprocessor.transform(X_test)

            logging.info('Preprocessor applied')

            # combining the transformed datasets and target variable

            train_arr = np.c_[input_feature_train_array,Y_train]
            test_arr = np.c_[input_feature_test_array]

            # saving the preprocessor object 

            save_object(file_path=self.config.preprocessing_obj_file, obj = preprocessor)
            logging.info('saved preprocessor')

            return train_arr, test_arr, self.config.preprocessing_obj_file

        except Exception as e:
            logging.error('Error in data transformation')
            raise customException(e,sys)



import os 
import sys 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense

from src.exception import customException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        if hasattr(obj, 'save') and callable(obj.save):  # For Keras models
            obj.save(file_path)  # Saves as .h5 automatically
        else:  # For scikit-learn models
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
                
    except Exception as e:
        raise customException(e, sys)
    

def plot_residuals(y_true, y_pred, model_name, save_dir='artifacts/residual_plots'):
    try:
        os.makedirs(save_dir, exist_ok=True)
        residuals = y_true - y_pred
        
        plt.figure(figsize=(8, 4))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'{model_name} - Residuals')
        plt.xlabel('Predicted RUL')
        plt.ylabel('Actual - Predicted')
        
        # Save as PNG instead of showing
        plot_path = os.path.join(save_dir, f'residuals_{model_name}.png')
        plt.savefig(plot_path)
        plt.close()  # Prevents memory leaks
        
    except Exception as e:
        raise customException(e,sys)



def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            if model_name == 'CNN-LSTM':
                # Custom handling for CNN-LSTM (no GridSearchCV)
                X_train_reshaped = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
                X_test_reshaped = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
                
                keras_model = Sequential([
                    LSTM(50),
                    Dense(1)
                ])
                keras_model.compile(optimizer='adam', loss='mse')
                keras_model.fit(X_train_reshaped, y_train, epochs=10, verbose=0)
                
                
                y_test_pred = keras_model.predict(X_test_reshaped).flatten()
                report[model_name] = r2_score(y_test, y_test_pred)
                
                # Store the keras model for later saving
                models[model_name] = keras_model

                plot_residuals(y_test, y_test_pred, model_name)
            else:
                # Original logic for scikit-learn models
                gs = GridSearchCV(model, param[model_name], cv=3, n_jobs=-1)
                gs.fit(x_train, y_train)
                model.set_params(**gs.best_params_)
                model.fit(x_train, y_train)
                y_test_pred = model.predict(x_test)
                report[model_name] = r2_score(y_test, y_test_pred)
                plot_residuals(y_test, y_test_pred, model_name)

        return report

    except Exception as e:
        raise customException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise customException(e, sys)



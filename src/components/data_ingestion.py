

import os 
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd 


from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import datatransformationconfig
from src.components.data_transformation import data_transformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class dataingestioncongif:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class dataingestion:
    def __init__(self):
        self.ingestion_config = dataingestioncongif()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method to read the data")

        try:
            train_set = pd.read_csv('Notebook/Data/train_FD001.txt', delim_whitespace = True, header = None)
            test_set = pd.read_csv('Notebook/Data/test_FD001.txt', delim_whitespace = True, header = None)
            
            logging.info("imported and read the train data file")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            logging.info('train data saved in csv in artifact')
            
            #logging.info('Train Test Split initiated')
            #train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)
            # (we were already provided with the two different datasets of train & test so, no need to split 
            # data into train & test datasets again)
            
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)


            logging.info('data ingestion completed.')

            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise customException(e,sys)

if __name__ == "__main__":
    obj=dataingestion()
    #obj.initiate_data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_trans = data_transformation()

    train_arr,test_arr,_ = data_trans.transform_data(train_data, test_data)

    model_trainer = ModelTrainer()
    Y_test = pd.read_csv('Notebook/Data/RUL_FD001.txt', delim_whitespace=True, header=None).values.flatten()
    print(model_trainer.initiaing_model_trainer(train_arr,test_arr,Y_test))
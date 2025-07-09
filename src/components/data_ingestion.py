import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

'''
from dataclasses import dataclass
dataclasses:
A built-in Python module (since Python 3.7) that simplifies the creation of classes used for storing data.

import dataclass:
You're importing the @dataclass decorator from the dataclasses module.
It automatically creates:
__init__() method
__repr__() method
__eq__() method
and others to reduce boilerplate code.
'''

'''
train_data_path:
The name of the variable. It will store the full path to your training dataset CSV file.

: str:
This is a type hint telling Python that this variable should hold a string.

=:
Assigns a default value to the attribute.

os.path.join('artifacts', 'train.csv'):

Joins folder 'artifacts' with file 'train.csv' using the OS-specific separator (e.g., / on Linux/macOS, \ on Windows).

This avoids hardcoding platform-dependent paths.

Result: 'artifacts/train.csv' (Linux/macOS) or 'artifacts\\train.csv' (Windows)

'''

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artificats','train.csv')
    test_data_path: str=os.path.join('artificats', 'test.csv')
    raw_data_path: str=os.path.join('artificats', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()      #The above 3 paths will be stored inside this variable

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df=pd.read_csv('../../notebook/data/stud.csv')
            logging.info('Read the dataset as DF')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train Test Split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of the data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path    #We are returning this coz the next step Data transformation can grab these infos and start the process.
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
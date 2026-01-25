import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    trainDataPath:str=os.path.join("artifacts","train.csv")
    testDataPath:str=os.path.join("artifacts","test.csv")
    rawDataPath:str=os.path.join("artifacts","raw.csv")
    

class DataIngestion:
    def __init__(self):
        self.ingestionConfig=DataIngestionConfig()
        
    def initiateDataIngestion(self):
        try:
            logging.info("Reading the DataSet")
        
            data=pd.read_csv("notebook/StudentsPerformance.csv")
            
            os.makedirs(os.path.dirname(self.ingestionConfig.trainDataPath),exist_ok=True)
            
            data.to_csv(self.ingestionConfig.rawDataPath,index=False,header=True)
            
            logging.info("splitting the dataset into train and test")
            
            trainSet,testSet=train_test_split(data,test_size=0.2,random_state=42)
            
            logging.info("writing the training data into the the csv")
            
            trainSet.to_csv(self.ingestionConfig.trainDataPath,index=False,header=True)
            
            logging.info("writing the test data into the csv")
            
            testSet.to_csv(self.ingestionConfig.testDataPath,index=False,header=True)
            
            return(
                self.ingestionConfig.trainDataPath,
                self.ingestionConfig.testDataPath
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    trainSet,testSet=obj.initiateDataIngestion()
    
    data_transformation=DataTransformation()
    trainFull,testFull,_=data_transformation.DataTransformationInitiate(trainSet,testSet)
    
    
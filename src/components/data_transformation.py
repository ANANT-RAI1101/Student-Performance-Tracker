import pandas as pd 
import numpy as np
from dataclasses import dataclass
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import saveObj


@dataclass

class DataTransformationConfig:
    preprocessorFilePath=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.dataTransformationConfig=DataTransformationConfig()
    
    def getTransformerObject(self):
       try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            
            numPipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            catPipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor=ColumnTransformer(
                [
                    ("numPipeline",numPipeline,numerical_columns),
                    ("catPipeline",catPipeline,categorical_columns)
                ]
            )
            return preprocessor
       except Exception as e:
           raise CustomException(e,sys)
    
    def DataTransformationInitiate(self,trainSet,testSet):
        try:
            logging.info("initiating data transformation")
            train_df=pd.read_csv(trainSet)
            test_df=pd.read_csv(testSet)
            
            logging.info("getting the preprocessor object")
            
            preprocessorObj=self.getTransformerObject()
            
            target_column_name="math score"
            
            inputFeaturesTrain=train_df.drop(columns=[target_column_name],axis=1)
            targetFeatureTrain=train_df[target_column_name]
            inputFeaturesTest=test_df.drop(columns=[target_column_name],axis=1)
            targetFeatureTest=test_df[target_column_name]
            
            trainInput=preprocessorObj.fit_transform(inputFeaturesTrain)
            testInput=preprocessorObj.transform(inputFeaturesTest)
            
            trainFull=np.c_[
                trainInput,np.array(targetFeatureTrain)
            ]
            
            testFull=np.c_[
                testInput,np.array(targetFeatureTest)
            ]
            
            logging.info("saving this preprocessed object")
            
            saveObj(
                filePath=self.dataTransformationConfig.preprocessorFilePath,
                obj=preprocessorObj
            )
            
            return(
                trainFull,
                testFull,
                self.dataTransformationConfig.preprocessorFilePath
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            
        
import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import loadObject

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            modelPath=os.path.join("artifacts","model.pkl")
            preprocessorPath=os.path.join("artifacts","preprocessor.pkl")
            logging.info("loading model and preprocessor")
            model=loadObject(fielPath=modelPath)
            preprocessor=loadObject(filePath=preprocessorPath)
            transformedData=preprocessor.transform(features)
            prediction=model.predict(transformedData)
            return prediction
        
        except Exception as e :
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def createDataFrame(self):
        try:
            data={
                "gender": self.gender,
                "race/ethnicity": self.race_ethnicity,
                "parental level of education": self.parental_level_of_education,
                "lunch": self.lunch,
                "test preparation course": self.test_preparation_course,
                "reading score": self.reading_score,
                "writing score": self.writing_score
            }
            return pd.DataFrame(data)
        
        except Exception as e:
            raise CustomException(e,sys)


        
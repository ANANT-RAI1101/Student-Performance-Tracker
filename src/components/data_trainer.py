import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import saveObj,evaluateModel

@dataclass
class ModelTrainerConfig:
    dataTrainerFilePath=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.modelTrainerConfig=ModelTrainerConfig()
    
    def modelTrainerInitiate(self,trainFull,testFull):
        try:
            logging.info("initiating the training ")
            logging.info("splitting the data for train test")
            X_train,y_train,X_test,y_test=(
                trainFull[:,:-1],
                trainFull[:,-1],
                testFull[:,:-1],
                testFull[:,-1]   
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256],
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
            }
            
            modelResults:dict=evaluateModel(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            bestScore=max(sorted(modelResults.values()))
            
            bestModelName=list(modelResults.keys())[
                list(modelResults.values()).index(bestScore)
            ]
            
            bestModel=models[bestModelName]
            
            if (bestScore<0.6):
                raise CustomException("evey model has score less than o.6")
            
            logging.info(f"saving the best model {bestModel}")
            
            saveObj(
                filePath=self.modelTrainerConfig.dataTrainerFilePath,
                obj=bestModel
            )
            
            prediction=bestModel.predict(X_test)
            
            r2Score=r2_score(y_test,prediction)
            
            return r2Score
        
        except Exception as e:
            raise CustomException(e,sys)
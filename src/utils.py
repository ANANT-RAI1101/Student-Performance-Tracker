# saveObj Function
import os
import pickle
import sys
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def saveObj(filePath,obj):
    try:
        dirPath=os.path.dirname(filePath)
        os.makedirs(dirPath,exist_ok=True)
        
        with open(filePath,"wb") as fileObj:
            pickle.dump(obj,fileObj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluateModel(X_train,y_train,X_test,y_test,models,params):
    try:
        result={}
        
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]
            
            tuner=GridSearchCV(model,para,cv=3)
            tuner.fit(X_train,y_train)
            
            model.set_params(**tuner.best_params_)
            model.fit(X_train,y_train)
            
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            
            trainR2Score=r2_score(y_train,y_train_pred)  
            testR2Score=r2_score(y_test,y_test_pred) 
            
            result[list(models.keys())[i]]=testR2Score    
            
        return result
    
    except Exception as e:
        raise CustomException(e,sys)     
            
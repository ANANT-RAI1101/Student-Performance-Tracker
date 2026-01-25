# saveObj Function
import os
import pickle
import sys
from src.exception import CustomException

def saveObj(filePath,obj):
    try:
        dirPath=os.path.dirname(filePath)
        os.makedirs(dirPath,exist_ok=True)
        
        with open(filePath,"wb") as fileObj:
            pickle.dump(obj,fileObj)
            
    except Exception as e:
        raise CustomException(e, sys)
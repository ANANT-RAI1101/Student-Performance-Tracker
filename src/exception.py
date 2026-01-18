import sys

def errorMessageDetails(error,errorDetails:sys):
    _,_,exc_tb=errorDetails.exc_info()
    fileName=exc_tb.tb_frame.f_code.co_filename
    errorMessage="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     fileName,exc_tb.tb_lineno,str(error))
    
    return errorMessage

class CustomException(Exception):
    def __init__(self, error,errorDetails:sys):
        super().__init__(error)
        self.error=errorMessageDetails(error,errorDetails=errorDetails)
        
    def __str__(self):
        return self.error
    
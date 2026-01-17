from setuptools import find_packages,setup
from typing import List
def get_requirements(filePath:str)->List[str]:
    requirements=[]
    with open(filePath) as fileObj:
        requirements=fileObj.readlines()
        requirements=[req.replace("/n","") for req in requirements]
        
        if("-e ." in requirements):
            requirements.remove("-e .")
            
    return requirements   

setup(
    name="StudentPerformanceTracker",
    version="0.0.1",
    author="Anant Rai",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)
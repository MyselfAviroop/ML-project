from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as file:
        requirements = file.readlines()
  
    requirements = [req.strip() for req in requirements if req.strip()]
    
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="ML_Project",
    version="0.1",
    author="Aviroop",
    author_email="ghoshaviroop542@gmail.com",
    description="A machine learning project",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

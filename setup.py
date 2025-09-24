from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """Read dependencies from requirements.txt"""
    with open(file_path) as f:
        requirements = f.readlines()
    # strip whitespace and ignore empty lines
    return [req.strip() for req in requirements if req.strip()]

setup(
    name="ML_Project",
    version="0.1",
    author="Aviroop",
    author_email="ghoshaviroop542@gmail.com",
    description="A machine learning project",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

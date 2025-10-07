from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """Read dependencies from requirements.txt"""
    requirements = []
    with open(file_path) as f:
        for line in f:
            req = line.strip()
            if req and req != "-e .":  # skip editable install
                requirements.append(req)
    return requirements

setup(
    name="ML_Project",   # 🚨 avoid spaces in project name
    version="0.1",
    author="Aviroop",
    author_email="ghoshaviroop542@gmail.com",
    description="A machine learning project",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

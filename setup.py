from setuptools import find_packages, setup
from typing import List


hy_e = '-e .'
def get_requirements(path:str)->List[str]:
    '''Returns a list of requirements'''
    requirement = []
    with open(path, 'r') as f:
        requirement = f.readlines()
        requirement = [req.replace('\n', '') for req in requirement]
        if hy_e in requirement:
            requirement.remove(hy_e)
    return requirement



setup(
    name = 'ML_Project',
    version='0.0.1',
    author='noumanirshad',
    author_email='noumanirshad564@gmail.com',
    description='Small ML Application',
    url='https://github.com/noumanirshad/ML_Project',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
)
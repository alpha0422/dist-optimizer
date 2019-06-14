from setuptools import setup, find_packages

setup(
    name='dist-optimizer',
    version='0.1',
    description='Distributed optimizer for PyTorch.',
    packages=find_packages(), 
    test_suite="tests",
)

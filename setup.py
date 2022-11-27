from setuptools import setup, find_packages


setup(
    name='kg_otto',
    description="Library for otto kaggle competition",
    author="AliExpress",
    packages=find_packages(where="."),
    include_package_data=True
)

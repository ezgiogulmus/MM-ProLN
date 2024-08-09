from setuptools import setup, find_packages


setup(
    name='mmproln',
    version='0.1.0',
    description='Lymph Node Metastasis Prediction in Prostate Cancer',
    url='https://github.com/ezgiogulmus/MM-ProLN',
    author='FEO',
    author_email='',
    license='MIT',
    packages=find_packages(exclude=['data', 'results']),
    install_requires=[
        "torch==1.13.1",
        "torchvision==0.14.1",
        "torcheval==0.0.6",
        "torchio==0.18.86",
        "numpy==1.23.4", 
        "pandas==1.4.3", 
        "openpyxl",
        "scikit-learn",
        "wandb",
        "matplotlib",
        "seaborn",
        "pyradiomics"
    ],

    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: MIT",
    ]
)
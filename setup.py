from setuptools import setup, find_packages

setup(
    name='ml_project',
    version='0.1.0',
    description='A machine learning project with explainable AI and visualization tools',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'pandas==1.5.3',
        'numpy==1.24.4',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'scipy==1.10.1',
        'plotly==5.15.0',
        'scikit-learn==1.2.2',
        'xgboost==1.7.6',
        'lightgbm==3.3.5',
        'shap==0.42.1',
        'lime==0.2.0.1'
    ],
    python_requires='>=3.8',
)

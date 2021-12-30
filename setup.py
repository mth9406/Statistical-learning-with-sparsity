from setuptools import setup, find_packages

setup(
    name= 'sparse_stats',
    version= '0.0.1',
    description= 'Implemenation of sparse statistical learning - Lasso and its generalization',
    author= 'SUNGWOO HUR',
    author_email= 'hursungwoo@postech.ac.kr',
    url= 'https://github.com/mth9406/Statistical-learning-with-sparsity.git',
    install_requires= ['numpy'],
    packages=find_packages(),
    zip_safe = False
)

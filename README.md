# Credit Card Fraud Detection Project

This repository contains code for a Credit Card Fraud Detection task using a PyTorch-based model. The dataset used for this project can be accessed on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Files

The important files in this repository are:

1. [kaggle_dataset_downloader.py](https://github.com/utkuozbudak/fraud_detection/blob/main/kaggle_dataset_downloader.py): This python script is used to download any specified Kaggle dataset using the Kaggle API. Please ensure that your Kaggle API credentials are present in the same directory as this file. The path where the dataset will be downloaded and the dataset to be downloaded can be specified as parameters.

2. [model_builder.py](https://github.com/utkuozbudak/fraud_detection/blob/main/model_builder.py): This file includes the PyTorch neural network architecture used in this project. 

3. [engine.py](https://github.com/utkuozbudak/fraud_detection/blob/main/engine.py): This file contains functions used to train the PyTorch model generated in `model_builder.py`. The functions included are for training step (`train_step`), testing step (`test_step`), and the overall training process (`train`).

4. [utils.py](https://github.com/utkuozbudak/fraud_detection/blob/main/utils.py): This file includes various helper functions used throughout the project, such as data visualization functions.

5. [fraud_detection.ipynb](https://github.com/utkuozbudak/fraud_detection/blob/main/fraud_detection.ipynb): This is the main Jupyter notebook used for data analysis, model initialization, training, and evaluation. All the above files are utilized within this notebook.

6. [data_setup.py](https://github.com/utkuozbudak/fraud_detection/blob/main/data_setup.py): This script prepares the dataset for the model. 

## Environment

This project was developed using the Google Colab environment. To replicate the results, it is recommended to run the Jupyter notebook in a similar environment.

## Setup

1. Clone this repository
2. Download the Kaggle credentials and place it in the same directory as `kaggle_dataset_downloader.py`
3. Run the `kaggle_dataset_downloader.py` script to download the required dataset
4. Open and run the `fraud_detection.ipynb` notebook.

Please refer to the individual scripts and the notebook for more detailed instructions and documentation.

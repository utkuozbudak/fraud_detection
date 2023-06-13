"""
This script facilitates the downloading of datasets from Kaggle. 
It's designed to function within a Google Colab environment but can also work in any Python-supported context. 
The user's Kaggle credentials are necessary for accessing Kaggle's API and should be provided in a 'kaggle.json' file.
"""

import os
import subprocess
import json
import zipfile

with open('kaggle.json') as f:
    kaggle_credentials = json.load(f)

os.environ['KAGGLE_USERNAME'] = kaggle_credentials['username']
os.environ['KAGGLE_KEY'] = kaggle_credentials['key']

def download_data(dataset: str, download_path: str):
  """
    Downloads and extracts a specified Kaggle dataset into a given directory.
    The function first verifies whether the dataset already exists in the target directory. 
    If it doesn't, the function downloads, extracts, and deletes the .zip file for clean-up. 

    Args:
        dataset (str): Identifier for the Kaggle dataset. 
            Example: 'mlg-ulb/creditcardfraud'
        download_path (str): Path to the directory where the dataset will be downloaded and extracted. 
            Example: '/content/drive/MyDrive/fraud_detection/data'    
    """
  command = ["pip", "install", "kaggle"]
  subprocess.run(command)
  
  zip_path = os.path.join(download_path, dataset.split('/')[-1] + '.zip')

  csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]

  # Check if the zip file or any .csv file exists in the directory
  if not os.path.exists(zip_path) and not csv_files:
      command = ["kaggle", "datasets", "download", "-d", dataset, "-p", download_path]
      subprocess.run(command, check=True)
    
      # Unzip the file
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
          zip_ref.extractall(download_path)
        
        # Delete the zip file
      os.remove(zip_path)

      print(f"Data successfully downloaded and extracted to: {download_path}")
  else:
      print(f"Data already exists in the specified directory: {download_path}")
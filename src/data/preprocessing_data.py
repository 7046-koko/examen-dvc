import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from check_structure import check_existing_file, check_existing_folder
import os

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Preprocessing and standardization of raw data')

    input_filepath = input_filepath
    input_filepath = f"{input_filepath}/raw.csv"
    output_filepath =  output_filepath

    process_data(input_filepath, output_filepath)

def process_data(input_filepath, output_filepath):
    # Import datasets
    df = import_dataset(input_filepath)
   
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Standardisation of the input data 
    X_train_scaled, X_test_scaled = standardization_data(X_train, X_test)

    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, y_train, y_test, output_filepath)

    # Save dataframes standardized to their respective output file paths
    save_dataframes_standardization(X_train_scaled, X_test_scaled, output_filepath)


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']

     # Delete the target and date are not numeric
    feats = df.drop(['date','silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def standardization_data(X_train,X_test):
    # Standardisation des données
    scaler = StandardScaler()
    X_train[X_train.columns] = pd.DataFrame(scaler.fit_transform(X_train), index = X_train.index)
    X_train_scaled = X_train[X_train.columns]
    X_test[X_test.columns] = pd.DataFrame(scaler.transform(X_test),index = X_test.index)
    X_test_scaled = X_test[X_test.columns] 
    return  X_train_scaled, X_test_scaled

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)


def save_dataframes_standardization(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes standardized  to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)            

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main("./data/raw","./data/processed_data")
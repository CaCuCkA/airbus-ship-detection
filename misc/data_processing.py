
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from misc.constants import CONST
from misc.segement_loader import SegmentationDataLoader


class DataPreparation:
    def __init__(self, csv_path: str, batch_size: int, random_state: int = CONST.RANDOM_STATE):
        """
        Initialize the DataPreparation class with the path to the CSV file, batch size for data generation,
        and a random state for reproducibility.

        Args:
            csv_path (str): The path to the CSV file containing the dataset.
            batch_size (int): The batch size to use for data generators.
            random_state (int): The seed for random operations to ensure reproducibility.
        """
        self.__csv_path = csv_path
        self.__batch_size = batch_size
        self.__random_state = random_state
        self.__df = self.__load_data()
        self.__ship_df = self.__prepare_ship_df()
        self.__train_ships = None
        self.__valid_ships = None


    def __load_data(self):
        """
        Load data from CSV and preprocess.

        Returns:
            pd.DataFrame: The loaded and initially processed DataFrame.
        """
        df = pd.read_csv(self.__csv_path)
        return df


    def __prepare_ship_df(self):
        """
        Prepare the ship DataFrame by processing 'EncodedPixels' and counting ships per image.

        Returns:
            pd.DataFrame: The DataFrame with processed 'EncodedPixels' and 'NumberOfShips'.
        """
        ship_df = self.__df.copy()
        ship_df['ShipAmount'] = ship_df['EncodedPixels'].notnull().astype(int)
        ship_df['EncodedPixels'] = ship_df['EncodedPixels'].replace(0, '')
        ship_df = ship_df.groupby('ImageId').sum().reset_index()
        return ship_df


    def __undersample_zeros(self, df):
        """
        Undersample images with zero ships to balance the dataset.

        Args:
            df (pd.DataFrame): The DataFrame to undersample.

        Returns:
            pd.DataFrame: The undersampled DataFrame.
        """
        zeros = df[df['ShipAmount'] == 0].sample(n=25_000, random_state=self.__random_state)
        nonzeros = df[df['ShipAmount'] != 0]
        return pd.concat((nonzeros, zeros))


    def split_data(self):
        """
        Split data into training and validation sets and apply undersampling.

        Returns:
            tuple: A tuple containing the training and validation SegmentationDataLoader.
        """
        train_ships, valid_ships = train_test_split(self.__ship_df, test_size=0.3, stratify=self.__ship_df['ShipAmount'])
        self.__train_ships = self.__undersample_zeros(train_ships)
        self.__valid_ships = self.__undersample_zeros(valid_ships)

        train_data = SegmentationDataLoader(np.array(self.__train_ships['ImageId']), self.__df, self.__batch_size)
        valid_data = SegmentationDataLoader(np.array(self.__valid_ships['ImageId']), self.__df, self.__batch_size)

        return train_data, valid_data
    

    def get_full_samples(self):
        return self.__train_ships, self.__valid_ships

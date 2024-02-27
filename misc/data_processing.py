
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_generator import SegmentationDataLoader


class DataPreparation:
    def __init__(self, csv_path: str, batch_size: int, random_state: int = 42):
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


    def __load_data(self):
        """
        Load data from CSV and preprocess.

        Returns:
            pd.DataFrame: The loaded and initially processed DataFrame.
        """
        df = pd.read_csv(self.__csv_path)
        df.fillna("", inplace=True)
        return df


    def __prepare_ship_df(self):
        """
        Prepare the ship DataFrame by processing 'EncodedPixels' and counting ships per image.

        Returns:
            pd.DataFrame: The DataFrame with processed 'EncodedPixels' and 'NumberOfShips'.
        """
        ship_df = self.__df.copy()
        ship_df['NumberOfShips'] = ship_df['EncodedPixels'].notnull().astype(int)
        ship_df['EncodedPixels'] = ship_df['EncodedPixels'].replace(0, '')
        ship_df = ship_df.groupby('ImageId').sum().reset_index()
        ship_df["EncodedPixels"] = ship_df["EncodedPixels"].apply(lambda x: x if x != 0 else "")
        return ship_df


    def __undersample_zeros(self, df):
        """
        Undersample images with zero ships to balance the dataset.

        Args:
            df (pd.DataFrame): The DataFrame to undersample.

        Returns:
            pd.DataFrame: The undersampled DataFrame.
        """
        zeros = df[df['NumberOfShips'] == 0].sample(n=25000, random_state=self.__random_state)
        nonzeros = df[df['NumberOfShips'] != 0]
        return pd.concat([nonzeros, zeros])


    def split_data(self):
        """
        Split data into training and validation sets and apply undersampling.

        Returns:
            tuple: A tuple containing the training and validation SegmentationDataLoader.
        """
        train_ships, valid_ships = train_test_split(self.__ship_df, test_size=0.3, stratify=self.__ship_df['NumberOfShips'], random_state=self.__random_state)
        train_ships = self.__undersample_zeros(train_ships)
        valid_ships = self.__undersample_zeros(valid_ships)

        train_data = SegmentationDataLoader(np.array(train_ships['ImageId']), self.__df, self.__batch_size)
        valid_data = SegmentationDataLoader(np.array(valid_ships['ImageId']), self.__df, self.__batch_size)

        return train_data, valid_data

"""
Persistence script
"""
from enum import Enum

import pandas as pd
from pandas.io.parsers import TextFileReader


class DataType(Enum):
    """
    Data Type class based on Enum
    """
    RAW: str = 'data/raw/'
    PROCESSED: str = 'data/processed/'


class PersistenceManager:
    """
    Persistence Manager class
    """

    @staticmethod
    def save_to_csv(
            data: list[dict] | pd.DataFrame, data_type: str,
            filename: str = 'data') -> bool:
        """
        Save list of dictionaries as csv file
        :param data: list of tweets as dictionaries
        :type data: list[dict]
        :param data_type: folder where data will be saved from DataType
        :type data_type: str
        :param filename: name of the file
        :type filename: str
        :return: confirmation for csv file created
        :rtype: bool
        """
        dataframe: pd.DataFrame
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            if not data:
                return False
            dataframe = pd.DataFrame(data)
        dataframe.to_csv(f'{str(data_type)}{filename}.csv', index=False)
        return True

    @staticmethod
    def load_from_csv(
            filename: str = 'processed_data.csv', chunk_size: int = 100000
    ) -> pd.DataFrame:
        """
        Load dataframe from CSV using chunk scheme
        :param filename: name of the file
        :type filename: str
        :param chunk_size: Number of chunks to split dataset
        :type chunk_size: int
        :return: dataframe retrieved from CSV after optimization with chunks
        :rtype: pd.DataFrame
        """
        dtypes: dict = {}
        dates = ['created', 'date']
        # TODO: define column names and its data type to add it to
        #  read_csv method and also the date columns to parse
        chunks: TextFileReader = pd.read_csv(
            filename, header=0, chunksize=chunk_size)
        chunk_list: list = []
        for chunk in chunks:
            chunk_list.append(chunk)
        dataframe: pd.DataFrame = pd.concat(chunk_list)
        return dataframe

    @staticmethod
    def save_to_pickle(
            dataframe: pd.DataFrame, filename: str = 'optimized_df.pkl'
    ) -> None:
        """
        Save dataframe to pickle file
        :param dataframe: dataframe
        :type dataframe: pd.DataFrame
        :param filename: name of the file
        :type filename: str
        :return: None
        :rtype: NoneType
        """
        dataframe.to_pickle(f'data/processed/{filename}')

    @staticmethod
    def load_from_pickle(filename: str = 'optimized_df.pkl') -> pd.DataFrame:
        """
        Load dataframe from Pickle file
        :param filename: name of the file to search and load
        :type filename: str
        :return: dataframe read from pickle
        :rtype: pd.DataFrame
        """
        dataframe: pd.DataFrame = pd.read_pickle(f'data/processed/{filename}')
        return dataframe

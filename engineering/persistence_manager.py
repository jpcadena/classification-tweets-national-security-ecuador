"""
Persistence script
"""
import json
from enum import Enum
from typing import Union

import pandas as pd
from pandas.io.parsers import TextFileReader

from core.config import ENCODING, CHUNK_SIZE


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
            data: Union[list[dict], pd.DataFrame], data_type: str,
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
        dataframe.to_csv(f'{str(data_type)}{filename}.csv', index=False,
                         encoding=ENCODING)
        return True

    @staticmethod
    def load_from_csv(
            filename: str = 'processed_data.csv', chunk_size: int = CHUNK_SIZE,
            dtypes: dict = None,
            parse_dates: list[str] = None,
            converters: dict = None
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
        # TODO: define column names and its data type to add it to
        #  read_csv method and also the date columns to parse
        text_file_reader: TextFileReader = pd.read_csv(
            filename, header=0, chunksize=chunk_size, encoding=ENCODING,
            dtype=dtypes,
            parse_dates=parse_dates,
            converters=converters
        )
        dataframe: pd.DataFrame = pd.concat(
            text_file_reader, ignore_index=True)
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

    @staticmethod
    def read_from_json(
            filename: str = 'references/related_words_users.json') -> dict:
        with open(filename, encoding=ENCODING) as file:
            data: dict[str, list[str]] = json.load(file)
        return data

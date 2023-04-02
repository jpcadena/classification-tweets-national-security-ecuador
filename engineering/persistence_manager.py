"""
Persistence script
"""
import json
import logging
from enum import Enum
from typing import Union

import pandas as pd
from pandas.io.parsers import TextFileReader

from core.config import ENCODING, CHUNK_SIZE

logger: logging.Logger = logging.getLogger(__name__)


class DataType(Enum):
    """
    Data Type class based on Enum
    """
    RAW: str = "data/raw/"
    PROCESSED: str = "data/processed/"
    REFERENCES: str = "references/"
    FIGURES: str = "reports/figures/"


class PersistenceManager:
    """
    Persistence Manager class
    """

    @staticmethod
    def save_to_csv(
            data: Union[list[dict], pd.DataFrame],
            data_type: DataType = DataType.PROCESSED,
            filename: str = "data.csv"
    ) -> bool:
        """
        Save list of dictionaries as csv file
        :param data: list of tweets as dictionaries
        :type data: list[dict]
        :param data_type: folder where data will be saved: RAW or
         PROCESSED
        :type data_type: DataType
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
        dataframe.to_csv(f"{str(data_type.value)}{filename}", index=False,
                         encoding=ENCODING)
        return True

    @staticmethod
    def load_from_csv(
            filename: str = "raw_tweets.csv",
            data_type: DataType = DataType.RAW, chunk_size: int = CHUNK_SIZE,
            dtypes: dict = None, parse_dates: list[str] = None
    ) -> pd.DataFrame:
        """
        Load dataframe from CSV using chunk scheme
        :param filename: name of the file
        :type filename: str
        :param data_type: Path where data will be saved: RAW or
         PROCESSED
        :type data_type: DataType
        :param chunk_size: Number of chunks to split dataset
        :type chunk_size: int
        :param dtypes: Dictionary of columns and datatypes
        :type dtypes: dict
        :param parse_dates: List of date columns to parse
        :type parse_dates: list[str]
        :return: dataframe retrieved from CSV after optimization with chunks
        :rtype: pd.DataFrame
        """
        filepath: str = f"{data_type.value}{filename}"
        text_file_reader: TextFileReader = pd.read_csv(
            filepath, header=0, chunksize=chunk_size, encoding=ENCODING,
            dtype=dtypes, parse_dates=parse_dates)
        dataframe: pd.DataFrame = pd.concat(
            text_file_reader, ignore_index=True)
        return dataframe

    @staticmethod
    def save_to_pickle(
            dataframe: pd.DataFrame, filename: str = "optimized_df.pkl",
            data_type: DataType = DataType.PROCESSED
    ) -> None:
        """
        Save dataframe to pickle file
        :param dataframe: dataframe
        :type dataframe: pd.DataFrame
        :param filename: name of the file
        :type filename: str
        :param data_type: Path where data will be saved: RAW or PROCESSED
        :type data_type: DataType
        :return: None
        :rtype: NoneType
        """
        filepath: str = f"{data_type.value}{filename}"
        dataframe.to_pickle(filepath)
        logger.info("Dataframe saved as pickle: %s", filepath)

    @staticmethod
    def load_from_pickle(
            filename: str = "optimized_df.pkl",
            data_type: DataType = DataType.PROCESSED
    ) -> pd.DataFrame:
        """
        Load dataframe from Pickle file
        :param filename: name of the file to search and load
        :type filename: str
        :param data_type: Path where data will be loaded from: RAW or
         PROCESSED
        :type data_type: DataType
        :return: dataframe read from pickle
        :rtype: pd.DataFrame
        """
        filepath: str = f"{data_type.value}{filename}"
        dataframe: pd.DataFrame = pd.read_pickle(filepath)
        logger.info("Dataframe loaded from pickle: %s", filepath)
        return dataframe

    @staticmethod
    def read_from_json(
            filename: str = "related_words_users.json",
            data_type: DataType = DataType.REFERENCES
    ) -> dict:
        """
        Read dataframe from JSON file
        :param filename: Name of the file to read
        :type filename: str
        :param data_type: Path where data will be loaded from. The
         default is REFERENCES
        :type data_type: DataType
        :return: Data read from file
        :rtype: dict[str, list[str]]
        """
        filepath: str = f"{data_type.value}{filename}"
        with open(filepath, encoding=ENCODING) as file:
            data: dict[str, list[str]] = json.load(file)
        return data

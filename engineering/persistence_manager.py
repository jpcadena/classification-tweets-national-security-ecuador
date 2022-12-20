"""
Persistence script
"""
from enum import Enum
import pandas as pd


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
    def save_to_file(
            data: list[dict] | pd.DataFrame, data_type: str, filename: str = 'data'
    ) -> bool:
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

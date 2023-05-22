"""
Data collection process
"""
import re
from datetime import date, datetime
from typing import Union, Optional, Callable, Any

import pandas as pd
import snscrape.modules.twitter as sn_twitter

from core.config import DATE_FORMAT


# TODO: define operations if data will be required to be written to disk.
#  If so, define persistence methods to save as csv or any file extension


def decode_tweet_to_json(obj: sn_twitter.Tweet) -> str:
    """
    Decode scraped tweet object to json string including datetime
    :param obj: Scraped Tweet object
    :type obj: Tweet
    :return: json string with tweet data
    :rtype: str
    """
    return (
        obj.strftime(DATE_FORMAT) if isinstance(obj, (date, datetime)) else
        obj.__dict__
    )


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date helper function
    :param date_str: The date as string
    :type date_str:str
    :return: The date as a datetime object
    :rtype: datetime
    """
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except ValueError:
        print(f"Error parsing date: {date_str}")
        return None


def str_to_datetime_values(data: dict) -> dict:
    """
    Function to cast string values to datetime from dictionary
    :param data: Tweet data
    :type data: dict
    :return: Tweet with datetime data type
    :rtype: dict
    """
    data["date"] = parse_date(data.get("date"))
    data["user"]["created"] = parse_date(data["user"].get("created"))
    data["quotedTweet"]["date"] = parse_date(data["quotedTweet"].get("date"))
    data["quotedTweet"]["user"]["created"] = parse_date(
        data["quotedTweet"]["user"].get("created")
    )
    return data


def camel_to_snake(name: str) -> str:
    """
    Convert Tweet keys from camelCase to snake_case
    :param name: name of the key in dictionary
    :type name: str
    :return: name converted into snake_case
    :rtype: str
    """
    edited_name: str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", edited_name).lower()


def nested_camel(data: Union[list, dict]) -> Union[list, dict]:
    """
    Apply camel_to_snake method to full dictionary
    :param data: data from Tweet
    :type data: Union[list, dict]
    :return: tweet with full snake_case
    :rtype: list or dict
    """
    if isinstance(data, list):
        return [nested_camel(i) if isinstance(i, (dict, list)) else i for i
                in data]
    return {
        camel_to_snake(a): nested_camel(b) if isinstance(b, (dict, list)) else
        b for a, b in data.items()
    }


def flatten(
        raw_tweet: dict, column_name: str, structure: list[str]
) -> pd.Series:
    """
    Flat method to nested dictionaries as dataframe column
    :param raw_tweet: Tweet data
    :type raw_tweet: dict
    :param column_name: Column name to apply function
    :type column_name: str
    :param structure: Nested dictionary structure keys
    :type structure: list[str]
    :return: Series created with prefix column_name + original name
    :rtype: pd.Series
    """
    data: dict = {}
    mod_keys = [column_name + "_" + sub for sub in structure]
    if not raw_tweet:
        data = dict.fromkeys(mod_keys, "")
    else:
        for key, value in raw_tweet.items():
            if isinstance(value, list):
                if isinstance(value[0], dict):
                    for k_s, v_s in value[0].items():
                        data[f"{column_name}_{key}_{k_s}"] = v_s
                else:
                    data[f"{column_name}_{key}"] = value
            else:
                data[f"{column_name}_{key}"] = value
    return pd.Series(data)


def get_nested_dict_structure(
        dataframe: pd.DataFrame, column: str
) -> list[str]:
    """
    Get nested structure from dictionary in Tweet dictionary
    :param dataframe: Tweet dataframe to search for nested dictionaries
    :type dataframe: pd.DataFrame
    :param column: Column name to find structure
    :type column: str
    :return: list of keys from nested dictionary
    :rtype: list[str]
    """
    column_list: list[dict] = list(
        dataframe[dataframe[column].notnull()][column].to_list()
    )
    structure: list[str] = []
    if column_list:
        structure = list(column_list[0].keys())
    return structure


def combine_flattened(
        dataframe: pd.DataFrame, column: str, func: Callable[..., Any],
        structure: list[str]
) -> pd.DataFrame:
    """
    Join Tweets dataframe with flatten dictionary as new Series columns
    :param dataframe: Original Tweets dataframe
    :type dataframe: pd.DataFrame
    :param column: Column name to apply func
    :type column: str
    :param func: function to use in apply pandas method
    :type func: Callable[..., Any]
    :param structure: structure of keys from nested dictionary
    :type structure: list[str]
    :return: Merged dataframe with new flatten columns
    :rtype: pd.DataFrame
    """
    merged_dataframe: pd.DataFrame = dataframe.join(
        dataframe[column].apply(func, args=(column, structure))
    ).drop([column], axis=1)
    return merged_dataframe

"""
Filter schema
"""
import json
from abc import ABC, abstractmethod

import snscrape.modules.twitter as sn_twitter
from schemas.specification import Specification


class Filter(ABC):
    """
    Filter class based on Abstract Base Class.
    """

    @abstractmethod
    def filter(self, spec: Specification, exclude: str = None) -> None:
        """
        Abstract method to filter
        :param spec: Object to filter by
        :type spec: Specification
        :param exclude: Text to exclude
        :type exclude: str
        :return: None
        :rtype: NoneType
        """


class BetterFilter(Filter):
    """
    Better Filter class based on Abstract Filter.
    """

    def filter(self, spec: Specification, exclude: str = None,
               limit: int = 100, func: callable = None) -> list[dict]:
        """
        Filter method inherited from Filter
        :param spec: Specification to use as filter
        :type spec: Specification
        :param exclude: word to exclude
        :type exclude: str
        :param limit: number of tweets to search
        :type limit: int
        :param func: function to apply as default for decode json
        :type func: function
        :return: list of raw tweets as dictionaries
        :rtype: list[dict]
        """
        raw_tweets: list[dict] = []
        query: str = spec.spec
        if exclude:
            query = query + ' -' + exclude
        for idx, tweet in enumerate(sn_twitter.TwitterSearchScraper(
                query).get_items()):
            if idx > limit:
                break
            full_tweet_dict: dict = json.loads(
                json.dumps(tweet.__dict__, default=func))
            raw_tweets.append(full_tweet_dict)
        return raw_tweets

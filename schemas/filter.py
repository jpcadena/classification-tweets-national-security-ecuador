"""
Filter schema
"""
import json
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any

import snscrape.modules.twitter as sn_twitter

from schemas.specification import Specification


class Filter(ABC):
    """
    Filter class based on Abstract Base Class.
    """

    @abstractmethod
    def filter(
            self, spec: Specification, exclude: Optional[str] = None
    ) -> None:
        """
        Abstract method to filter
        :param spec: Object to filter by
        :type spec: Specification
        :param exclude: Text to exclude
        :type exclude: Optional[str]
        :return: None
        :rtype: NoneType
        """


class BetterFilter(Filter):
    """
    Better Filter class based on Abstract Filter.
    """

    def filter(
            self,
            spec: Specification,
            exclude: Optional[str] = None,
            limit: int = 100,
            func: Optional[Callable[..., Any]] = None,
    ) -> list[dict]:
        """
        Filter method inherited from Filter
        :param spec: Specification to use as filter
        :type spec: Specification
        :param exclude: word to exclude
        :type exclude: Optional[str]
        :param limit: number of tweets to search
        :type limit: int
        :param func: function to apply as default for decode json
        :type func: Optional[Callable[..., Any]]
        :return: list of raw tweets as dictionaries
        :rtype: list[dict]
        """
        raw_tweets: list[dict] = []
        query: str = spec.spec
        if exclude:
            query = query + " -" + exclude

        # tweet_test: sn_twitter.TwitterTweetScraper = \
        #     sn_twitter.TwitterTweetScraper(
        #         1620587742588551169,
        #         mode=sn_twitter.TwitterTweetScraperMode.SINGLE)
        # tweet_retrieve: list[sn_twitter.Tweet] = list(tweet_test.get_items())
        # for single_tweet in tweet_retrieve:
        #     tweet_dict: dict = json.loads(single_tweet.json())

        for idx, tweet in enumerate(
                sn_twitter.TwitterSearchScraper(query).get_items()):
            if idx > limit:
                break
            full_tweet_dict: dict = json.loads(json.dumps(
                tweet.__dict__, default=func))
            raw_tweets.append(full_tweet_dict)
        return raw_tweets

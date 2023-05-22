"""
Extraction script for Engineering module
"""
import nltk

from engineering.snscrape_collection import decode_tweet_to_json
from schemas.filter import BetterFilter
from schemas.specification import SenderSpecification, TextSpecification


def collect_tweets(
        json_file: dict[str, list[str]]
) -> tuple[list[dict], list[dict]]:
    """
    Collect tweets from multiple users and topics
    :param json_file: The json file containing the users and topics
    :type json_file: dict[str, list[str]]
    :return: The list of tweets and the list of additional tweets
    :rtype: tuple[list[dict], list[dict]]
    """
    print("collect_tweets")
    insecurity_words: list[str] = json_file.get("words")
    users: list[str] = json_file.get("users")
    better_filter: BetterFilter = BetterFilter()
    tweets_from_users: list[dict] = collect_tweets_from_users(
        users, better_filter)
    tweets_collected: list[dict] = collect_tweets_containing_words(
        insecurity_words, better_filter
    )
    additional_tweets: list[dict] = collect_additional_tweets(better_filter)
    tweets_collected.extend(tweets_from_users)
    tweets_collected.extend(additional_tweets)
    return tweets_collected, additional_tweets


def collect_tweets_from_users(
        users: list[str], better_filter: BetterFilter
) -> list[dict]:
    """
    Collect tweets from users list
    :param users: The list of users
    :type users: list[str]
    :param better_filter: The BetterFilter instance
    :type better_filter: BetterFilter
    :return: The list of raw tweets as dictionaries
    :rtype: list[dict]
    """
    print("collect_tweets_from_users")
    tweets_from_users: list[dict] = []
    for user in users:
        sender_spec: SenderSpecification = SenderSpecification(user)
        tweets_from_users.extend(
            better_filter.filter(
                sender_spec, limit=150, func=decode_tweet_to_json)
        )
    return tweets_from_users


def collect_tweets_containing_words(
        insecurity_words: list[str], better_filter: BetterFilter
) -> list[dict]:
    """
    Collect tweets from specific words for insecurity topic
    :param insecurity_words: The list of insecurity related words
    :type insecurity_words: list[str]
    :param better_filter: The BetterFilter instance
    :type better_filter: BetterFilter
    :return: The list of raw tweets as dictionaries
    :rtype: list[dict]
    """
    print("collect_tweets_containing_words")
    tweets_collected: list[dict] = []
    for word in insecurity_words:
        if "" "" in word:
            continue
        text_spec: TextSpecification = TextSpecification(word)
        tweets_collected.extend(
            better_filter.filter(
                text_spec, limit=350, func=decode_tweet_to_json)
        )
    return tweets_collected


def collect_additional_tweets(better_filter: BetterFilter) -> list[dict]:
    """
    Collect additional tweets from specific words non insecurity related
    :param better_filter: The BetterFilter instance
    :type better_filter: BetterFilter
    :return: The list of raw tweets as dictionaries
    :rtype: list[dict]
    """
    print("collect_additional_tweets")
    additional_tweets: list[dict] = []
    additional_words: list[str] = ["turismo", "gastronomia", "futbol"]
    for add_word in additional_words:
        new_spec: TextSpecification = TextSpecification(add_word)
        additional_tweets.extend(
            better_filter.filter(
                new_spec, limit=1500, func=decode_tweet_to_json)
        )
    return additional_tweets


def get_stop_words(stopwords_file: dict[str, list[str]]) -> list[str]:
    """
    Get stop words list by combining Spanish stop words from NLTK and
     stop words specified in a JSON file
    :param stopwords_file: Dictionary of stop words from JSON file
    :type stopwords_file: dict[str, list[str]]
    :return: A list of stop words
    :rtype: list[str]
    """
    exclude_words: list[str] = stopwords_file.get("spanish")
    stop_words: list[str] = nltk.corpus.stopwords.words("spanish")
    stop_words.extend(exclude_words)
    stop_words = list(set(stop_words))
    return stop_words

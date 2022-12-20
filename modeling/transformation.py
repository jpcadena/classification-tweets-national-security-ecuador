"""
Transformation including: Scaling, Decomposition, Aggregation
"""
from nltk import WordNetLemmatizer


def normalization(tweet_list: list[str]) -> list[str]:
    """
    Lexicon normalization for words in different conjugations
    :param tweet_list: clean tweet as list
    :type tweet_list: list[str]
    :return: normalized tweet with unique words
    :rtype: list[str]
    """
    lem: WordNetLemmatizer = WordNetLemmatizer()
    normalized_tweet: list[str] = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word, 'v')
        normalized_tweet.append(normalized_text)
    return normalized_tweet

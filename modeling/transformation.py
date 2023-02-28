"""
Transformation including: Scaling, Decomposition, Aggregation
"""
import pandas as pd
from gensim.utils import simple_preprocess
from nltk import WordNetLemmatizer
from scipy.sparse import csr_matrix
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.feature_extraction.text import CountVectorizer


def normalization(tweet_list: list[str]) -> list[str]:
    """
    Lexicon normalization for words in different conjugations
    :param tweet_list: List of cleaned tweets
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


def remove_stopwords_and_tokenize(
        text: str, stop_words: list[str]) -> list[str]:
    """
    Removes stopwords from a string of text and tokenizes it
    :param text: The text to process
    :type text: str
    :param stop_words: A list of stopwords to remove
    :type stop_words: list[str]
    :return: A list of tokens without stopwords
    :rtype: list[str]
    """
    return [w for w in simple_preprocess(text) if
            w not in stop_words and len(w) >= 3]


def get_ngram_counts(tweet: str, stop_words: list[str]) -> dict[str, int]:
    """
    Calculates the count of n-grams in a tweet
    :param tweet: The tweet to process
    :type tweet: str
    :param stop_words: A list of stopwords to remove
    :type stop_words: list[str]
    :return: A dictionary with the count of each n-gram.
    :rtype: dict[str, int]
    """
    token_counts_matrix: CountVectorizer = CountVectorizer(
        stop_words=stop_words, ngram_range=(1, 3))
    vocabulary: dict
    ngrams_count: dict = {}
    try:
        doc_term_matrix = token_counts_matrix.fit_transform(tweet.split('\n'))
        vocabulary = token_counts_matrix.vocabulary_
        ngrams_count = dict(zip(vocabulary.keys(),
                                doc_term_matrix.sum(axis=0).tolist()[0]))
    except (FitFailedWarning, NotFittedError) as exc:
        print(exc)
    return ngrams_count


def text_to_bow(
        dataframe: pd.DataFrame, column_name: str
) -> tuple[csr_matrix, CountVectorizer]:
    """
    Convert text in a dataframe column to a bag of words representation
    :param dataframe: The dataframe containing the text column to
     convert
    :type dataframe: pd.DataFrame
    :param column_name: The name of the column containing the text to
     convert
    :type column_name: str
    :return: A tuple containing the bag of words matrix and the
     Count Vectorizer used to create it.
    :rtype: tuple[csr_matrix, CountVectorizer]
    """
    c_vec: CountVectorizer = CountVectorizer()
    c_bow: csr_matrix = c_vec.fit_transform(dataframe[column_name])
    return c_bow, c_vec

"""
Preprocessing section including: Formatting, Cleaning, Anonymization, Sampling
"""
import re
import string
from re import Pattern

import es_core_news_sm
import numpy as np
import pandas as pd
import spacy
from matplotlib import pyplot as plt
from nltk import WhitespaceTokenizer, RegexpTokenizer, WordNetLemmatizer, \
    TweetTokenizer
from nltk.corpus import stopwords
from numpy import float16
from sklearn.neighbors import LocalOutlierFactor
from textblob import TextBlob

from core.config import NUMERICS
from engineering.persistence_manager import DataType

spacy.prefer_gpu()
STOPWORDS_PATTERN: str = r"[^\W\d]*$"
nlp = es_core_news_sm.load()
tokenizer: RegexpTokenizer = RegexpTokenizer(r"\w +")
word_net_lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("spanish"))
punctuation = list(
    string.punctuation
)  # already taken care of with the cleaning function.
stop_words.update(punctuation)
whitespace_tokenizer: WhitespaceTokenizer = WhitespaceTokenizer()

# TODO: add anonymization functions for user_id, tweet_id, etc.,
#  add more cleaning functions.

tweet_tokenizer: TweetTokenizer = TweetTokenizer()


def tokenize(tweet: str) -> str:
    """
    Tokenize the input tweet text
    :param tweet: The tweet text to be tokenized
    :type tweet: str
    :return: The tokenized tweet text as a string
    :rtype: str
    """
    tokenized_tweet: list[str] = tweet_tokenizer.tokenize(tweet)
    return ", ".join(tokenized_tweet)


def lof_observation(
        dataframe: pd.DataFrame, data_type: DataType = DataType.FIGURES
) -> pd.DataFrame:
    """
    This function identifies outliers with LOF method
    :param dataframe: Dataframe containing data
    :type dataframe: pd.DataFrame
    :param data_type: folder where data will be saved. Defaults to FIGURES
    :type data_type: DataType
    :return: clean dataframe without outliers from LOF
    :rtype: pd.DataFrame
    """
    df_num_cols: pd.DataFrame = dataframe.select_dtypes(include=NUMERICS)
    df_outlier: pd.DataFrame = df_num_cols.astype("float64")
    clf: LocalOutlierFactor = LocalOutlierFactor(
        n_neighbors=20, contamination=0.1)
    clf.fit_predict(df_outlier)
    df_scores = clf.negative_outlier_factor_
    scores_df: pd.DataFrame = pd.DataFrame(np.sort(df_scores))
    scores_df.plot(
        stacked=True,
        xlim=[0, 20],
        color="r",
        title="Visualization of outliers according to the LOF method",
        style=".-",
    )
    plt.savefig(f"{data_type.value}outliers.png")
    plt.show()
    th_val = np.sort(df_scores)[2]
    outliers: bool = df_scores > th_val
    cleaned_df: pd.DataFrame = dataframe.drop(df_outlier[~outliers].index)
    print(cleaned_df.shape)
    return cleaned_df


def clear_outliers(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This function remove the outliers from specific column
    :param dataframe: Dataframe containing data
    :type dataframe: pd.DataFrame
    :param column: Column name
    :type column: str
    :return: clean dataframe from outliers using IQR
    :rtype: pd.DataFrame
    """
    first_quartile: float = dataframe[column].quantile(0.25)
    third_quartile: float = dataframe[column].quantile(0.75)
    iqr: float = third_quartile - first_quartile
    lower: float = first_quartile - 1.5 * iqr
    upper: float = third_quartile + 1.5 * iqr
    print(f"{column}- Lower score: ", lower, "and upper score: ", upper)
    df_outlier = dataframe[column][(dataframe[column] > upper)]
    print(df_outlier)
    return dataframe


def form_sentence(tweet: str) -> str:
    """
    Function to clean hashtags, mentions and punctuation
    :param tweet: raw tweet
    :type tweet: str
    :return: clean tweet
    :rtype: str
    """
    tweet_blob: TextBlob = TextBlob(text=tweet)
    clean_tweet: str = " ".join(tweet_blob.words)
    return clean_tweet


def clean_stopwords(tweet: str) -> list[str]:
    """
    Remove stopwords from spanish to a single tweet
    :param tweet: Text from a single tweet
    :type tweet: str
    :return: clean tweet as single words in a list of words
    :rtype: list[str]
    """
    tweet_list: list[str] = [ele for ele in tweet.split() if ele != "user"]
    clean_tokens: list[str] = [t for t in tweet_list if
                               re.match(STOPWORDS_PATTERN, t)]
    clean_s: str = "" "".join(clean_tokens)
    clean_mess: list[str] = [
        word
        for word in clean_s.split()
        if word.lower() not in stopwords.words("spanish")
    ]
    return clean_mess


def str_to_category(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Cast object (str) columns from dataframe to categorical
    :param dataframe: raw dataframe to optimize
    :type dataframe: pd.DataFrame
    :return: converted dataframe
    :rtype: pd.DataFrame
    """
    # if more than 50% of values in an object(str) column are unique,
    # its more feasible to continue using object rather than converting
    # them to category.
    df_object = dataframe.select_dtypes(include=["object"]).copy()
    for column in df_object:
        if len(df_object[column].unique()) < df_object[column].size / 2:
            df_object[column] = df_object[column].astype("category")
    dataframe[df_object.columns] = df_object
    return dataframe


def furnished(text: str) -> str:
    """
    Clean and preprocess the text by tokenizing, removing stop words
     and lemmatizing.
    :param text: The text to be cleaned and preprocessed
    :type text: str
    :return: The preprocessed text as a string
    :rtype: str
    """
    final_text: list[str] = []
    word: str = ""
    for i in whitespace_tokenizer.tokenize(text):
        if i.lower() not in stop_words:
            word = word_net_lemmatizer.lemmatize(i)
        final_text.append(word.lower())
    return " ".join(final_text)


def twitter_text_cleaning(text: str):
    """
    Clean a text message that might have @mentions, "#" symbol or URLs
    :param text: The text message to clean
    :type text: str
    :return: The cleaned text message without @mentions, "#" and URLs
    :rtype: str
    """
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"@[A-Za-zA-Z0-9]+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"@[A-Za-z]+", "", text)
    text = re.sub(r"@[-)]+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"https?\/\/\S+", "", text)
    text = re.sub(r"http?\/\/\S+", "", text)
    text = re.sub(r"https?\/\/.*[\r\n]*", "", text)
    text = re.sub(r"^https?\/\/.*[\r\n]*", "", text)
    text = re.sub(r"^https?:\/\/.*[\r\n]*", "", text)
    return text


def remove_emoji(text: str) -> str:
    """
    Remove emojis from a text
    :param text: The text to remove emojis from
    :type text: str
    :return: The input text with emojis removed.
    :rtype: str
    """
    emoji_pattern: Pattern[str] = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U0001f900-\U0001f9ff"  # Unicode 9.0 emojis
        "\U0001f980-\U0001f9ff"  # Unicode 10.0 emojis
        "\U0001fa80-\U0001faff"  # Unicode 11.0 emojis
        "\U0001fbc0-\U0001fbc9"  # Unicode 12.0 emojis
        "\U0001fcc0-\U0001fcc9"  # Unicode 13.0 emojis
        "\U0001fcd0-\U0001fcd9"  # Unicode 14.0 emojis
        "\U0001fdd0-\U0001fdd9"  # Unicode 15.0 emojis
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def remove_punc(message):
    """
    Remove punctuation from the given message.
    :param message: Text message that might have punctuations
    :type message: str
    :return: Message without punctuations
    :rtype: str
    """
    return "".join([char for char in message if char not in
                    string.punctuation])


def jaccard_similarity(query: list[str], document: list[str]) -> float16:
    """
    Calculates the Jaccard similarity between two sets of strings
    :param query: A list of strings representing a set of items
    :type query: list[str]
    :param document: A list of strings representing another set of
     items
    :type document: list[str]
    :return: The Jaccard similarity score between the two sets of items
    :rtype: float16
    """
    intersection: set[str] = set(query).intersection(set(document))
    union: set[str] = set(query).union(set(document))
    return float16(len(intersection) / len(union))


def get_scores(group: list[str], tweets: list[list[str]]) -> list[float16]:
    """
    Calculates the Jaccard similarity scores between a group of
     keywords and a list of tweets.
    :param group: A list of keywords
    :type group: list[str]
    :param tweets: A list of tweets represented as a list of words
    :type tweets: list[list[str]]
    :return: A list of Jaccard similarity scores between the group and
     each tweet
    :rtype: list[float16]
    """
    scores: list[float16] = []
    for tweet in tweets:
        score: float16 = jaccard_similarity(group, tweet)
        scores.append(score)
    return scores

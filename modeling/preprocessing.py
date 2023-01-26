"""
Preprocessing section including: Formatting, Cleaning, Anonymization, Sampling
"""
import re
import string

import es_core_news_sm
import numpy as np
import pandas as pd
import spacy
from matplotlib import pyplot as plt
from nltk import WhitespaceTokenizer, RegexpTokenizer, WordNetLemmatizer, \
    TweetTokenizer
from nltk.corpus import stopwords
from sklearn.neighbors import LocalOutlierFactor
from textblob import TextBlob

spacy.prefer_gpu()
STOPWORDS_PATTERN: str = r'[^\W\d]*$'
nlp = es_core_news_sm.load()
tokenizer = RegexpTokenizer(r'\w +')
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('spanish'))
punctuation = list(
    string.punctuation)  # already taken care of with the cleaning function.
stop.update(punctuation)
w_tokenizer = WhitespaceTokenizer()

# TODO: add anonymization functions for user_id, tweet_id, etc.,
#  add more cleaning functions.

tweet_tokenizer: TweetTokenizer = TweetTokenizer()


def tokenize(tweet: str) -> str:
    tweet_tokenizer.tokenize(tweet)


def downcast_type(dataframe: pd.DataFrame):
    """
    Optimization of numeric columns by down-casting its datatype
    :param dataframe: dataframe to optimize
    :type dataframe: pd.DataFrame
    :return: optimized dataframe
    :rtype: pd.DataFrame
    """
    numerics: list[str] = [
        'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32',
        'int64']
    numeric_ranges: list[tuple] = [
        (0, 255), (0, 65535), (0, 4294967295), (0, 18446744073709551615),
        (-128, 127), (-32768, 32767), (-2147483648, 2147483647),
        (-18446744073709551616, 18446744073709551615)]
    df_num_cols: pd.DataFrame = dataframe.select_dtypes(include=numerics)
    for column in df_num_cols:
        new_type: str = numerics[numeric_ranges.index(
            [num_range for num_range in numeric_ranges if
             df_num_cols[column].min() > num_range[0] and
             num_range[1] <= df_num_cols[column].max()][0])]
        df_num_cols[column] = df_num_cols[column].apply(
            pd.to_numeric, downcast=new_type)  # check map for Pd.Series
    dataframe[df_num_cols.columns] = df_num_cols
    return dataframe


def lof_observation(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function identifies outliers with LOF method
    :param dataframe: Dataframe containing data
    :type dataframe: pd.DataFrame
    :return: clean dataframe without outliers from LOF
    :rtype: pd.DataFrame
    """
    numerics: list[str] = [
        'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num_cols: pd.DataFrame = dataframe.select_dtypes(include=numerics)
    df_outlier: pd.DataFrame = df_num_cols.astype("float64")
    clf: LocalOutlierFactor = LocalOutlierFactor(
        n_neighbors=20, contamination=0.1)
    clf.fit_predict(df_outlier)
    df_scores = clf.negative_outlier_factor_
    scores_df: pd.DataFrame = pd.DataFrame(np.sort(df_scores))
    scores_df.plot(stacked=True, xlim=[0, 20], color='r',
                   title='Visualization of outliers according to the LOF '
                         'method', style='.-')
    plt.savefig('reports/figures/outliers.png')
    plt.show()
    th_val = np.sort(df_scores)[2]
    outliers: bool = df_scores > th_val
    dataframe: pd.DataFrame = dataframe.drop(df_outlier[~outliers].index)
    print(dataframe.shape)
    return dataframe


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
    clean_tweet: str = ' '.join(tweet_blob.words)
    return clean_tweet


def clean_stopwords(tweet: str) -> list[str]:
    """
    Remove stopwords from spanish to a single tweet
    :param tweet: tweet
    :type tweet: str
    :return: clean tweet as single words in a list
    :rtype: list[str]
    """
    tweet_list: str = [ele for ele in tweet.split() if ele != 'user']
    clean_tokens: list[str] = [t for t in tweet_list if re.match(
        STOPWORDS_PATTERN, t)]
    clean_s: str = ' '.join(clean_tokens)
    clean_mess: list[str] = [word for word in clean_s.split() if word.lower()
                             not in stopwords.words('spanish')]
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
    df_object = dataframe.select_dtypes(include=['object']).copy()
    for column in df_object:
        if len(df_object[column].unique()) < df_object[column].size / 2:
            df_object[column] = df_object[column].astype('category')
    dataframe[df_object.columns] = df_object
    return dataframe


def furnished(text) -> str:
    print(type(text))
    final_text: list[str] = []
    word: str = ''
    for i in w_tokenizer.tokenize(text):
        if i.lower() not in stop:
            word = lemmatizer.lemmatize(i)
        final_text.append(word.lower())
    return " ".join(final_text)

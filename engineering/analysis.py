"""
First Analysis script
"""
import pandas as pd

# TODO: Get dataframe to analyze and set it as tweets_df.
tweets_df: pd.DataFrame = pd.DataFrame()

interest_column: str = 'sentiment'
print(tweets_df.head())
print(tweets_df.shape)
print(tweets_df.dtypes)
print(tweets_df.info())
print(tweets_df.describe(include='all', datetime_is_numeric=True))
missing_values = (tweets_df.isnull().sum())
print(missing_values[missing_values > 0])
print(missing_values[missing_values > 0] / tweets_df.shape[0] * 100)

# Identifying Class Imbalance in Exited
print(tweets_df[interest_column].value_counts())
print(tweets_df[interest_column].unique())
print(tweets_df[interest_column].value_counts(normalize=True) * 100)

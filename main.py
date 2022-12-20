"""
Main script for ML project
"""
import pandas as pd
from engineering.persistence_manager import PersistenceManager, DataType
from engineering.snscrape_collection import decode_tweet_to_json, flatten, \
    str_to_datetime_values, nested_camel, get_nested_dict_structure, \
    combine_flattened
from schemas.filter import BetterFilter
from schemas.specification import SenderSpecification, ReceiverSpecification

better_filter: BetterFilter = BetterFilter()
sender_spec: SenderSpecification = SenderSpecification('TDataScience')
receiver_spec: ReceiverSpecification = ReceiverSpecification('JuanPabloCadena')
tweets_collected: list[dict] = better_filter.filter(
    sender_spec, limit=5, func=decode_tweet_to_json)
# raw_saved: bool = PersistenceManager.save_to_file(
#     tweets_collected, DataType.RAW.value, 'my_raw_tweets')
# if raw_saved:
#     print('raw tweets saved!')

# TODO: encapsulate cleaning process into simple functions

clean_tweets: list[dict] = []
for element in tweets_collected:
    print(f"Tweet id: {element['id']}")
    element: dict = str_to_datetime_values(element)
    element: dict = nested_camel(element)
    clean_tweets.append(element)
tweets_df: pd.DataFrame = pd.DataFrame(clean_tweets)
user_structure: list[str] = get_nested_dict_structure(tweets_df, 'user')
quoted_tweet_structure: list[str] = get_nested_dict_structure(
    tweets_df, 'quoted_tweet')
tweets_df_w_user: pd.DataFrame = combine_flattened(
    tweets_df, 'user', flatten, user_structure)
clean_tweets_df: pd.DataFrame = combine_flattened(
    tweets_df_w_user, 'quoted_tweet', flatten, quoted_tweet_structure)
clean_tweets_df.drop(
    ['source', 'source_url', 'cashtags', 'quoted_tweet_source',
     'quoted_tweet_source_url', 'quoted_tweet_cashtags'], axis=1, inplace=True,
    errors='ignore')
print(clean_tweets_df)
# processed_saved: bool = PersistenceManager.save_to_file(
#     clean_tweets_df, DataType.PROCESSED.value, 'processed_tweets')
# if processed_saved:
#     print('clean tweets saved!')

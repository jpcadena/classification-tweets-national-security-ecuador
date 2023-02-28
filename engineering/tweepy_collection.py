"""
Tweepy data collection
"""
import os

from dotenv import load_dotenv

# TODO: Wait for Twitter API approval for Elevated access to be approved

load_dotenv()
consumer_key: str = os.environ["API_KEY"]
consumer_secret: str = os.environ["API_KEY_SECRET"]
access_token: str = os.environ["ACCESS_TOKEN"]
access_token_secret: str = os.environ["ACCESS_TOKEN_SECRET"]
bearer_token: str = os.environ["BEARER_TOKEN"]

# authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)
# authenticate.set_access_token(access_token, access_token_secret)
# api = tweepy.API(authenticate, wait_on_rate_limit=True)
#
# # Client seems to be the best option for new API version
#
# client = tweepy.Client(
#     bearer_token=bearer_token,
#     consumer_key=consumer_key,
#     consumer_secret=consumer_secret, access_token=access_token,
#     access_token_secret=access_token_secret,
#     wait_on_rate_limit=True
# )
#
# query: str = 'Seguridad nacional'
#
# tweets = client.search_all_tweets(
#     query=query, max_results=100,
#     tweet_fields=['context_annotations', 'created_at'])
#
# for tweet in tweets.data:
#     print("tweet", type(tweet))
#     print(tweet.text)
#     if len(tweet.context_annotations) > 0:
#         print(tweet.context_annotations)
#
# # Another way can be Cursor but got same issue with privileges.
#
# keywords: str = 'World Cup Qatar 2022'
# limit: int = 10
# tweets = tweepy.Cursor(
#     api.search_tweets, q=keywords, count=5, tweet_mode='extended').items(
#     limit)
# columns: list[str] = ['User', 'Tweet']
# data: list = []
# for tweet in tweets:
#     data.append([tweet.user.screen_name, tweet.full_text])
# dataframe: pd.DataFrame = pd.DataFrame(data, columns=columns)
# print(dataframe)

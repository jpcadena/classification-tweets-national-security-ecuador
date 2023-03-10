"""
Main script for ML project
"""
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import uint32, float16, uint8

from core import logging_config
from engineering.analysis import silhouette_scores, \
    latent_dirichlet_allocation, kmeans_clustering
from engineering.extraction import collect_tweets, get_stop_words
from engineering.persistence_manager import PersistenceManager, DataType
from engineering.snscrape_collection import flatten, \
    str_to_datetime_values, nested_camel, get_nested_dict_structure, \
    combine_flattened
from engineering.visualization import elbow_method, visualize_clusters
from modeling.preprocessing import twitter_text_cleaning, remove_emoji, \
    remove_punc, get_scores
from modeling.transformation import remove_stopwords_and_tokenize, \
    get_ngram_counts, text_to_bow
from models.models import iterate_models

logging_config.setup_logging()
logger: logging.Logger = logging.getLogger(__name__)
plt.rcParams.update({'figure.max_open_warning': 0})

json_file: dict[str, list[str]] = PersistenceManager.read_from_json()
tweets_collected, additional_tweets = collect_tweets(json_file)

raw_tweets_df: pd.DataFrame = pd.DataFrame(tweets_collected)

raw_saved: bool = PersistenceManager.save_to_csv(
    raw_tweets_df, DataType.RAW, 'raw_tweets.csv')
if raw_saved:
    print('raw tweets saved!')

# TODO: encapsulate cleaning process into simple functions

clean_tweets: list[dict] = []
for element in tweets_collected:
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
num_cols: list[str] = [
    'reply_count', 'retweet_count', 'like_count', 'quote_count',
    'view_count',
    'user_followers_count', 'user_friends_count', 'user_favourites_count']
for c in clean_tweets_df.columns:
    if c in num_cols:
        clean_tweets_df[c] = clean_tweets_df[c].fillna(uint32(0))
        clean_tweets_df[c] = clean_tweets_df[c].astype(uint32)
clean_tweets_df['date'] = pd.to_datetime(clean_tweets_df['date'])
clean_tweets_df['user_created'] = pd.to_datetime(
    clean_tweets_df['user_created'])

stopwords_file: dict[str, list[str]] = PersistenceManager.read_from_json(
    'stop_words.json')
stop_words: list[str] = get_stop_words(stopwords_file)

snscrape_columns: list[str] = stopwords_file.get('tweets')
clean_tweets_df.drop(snscrape_columns, axis=1, inplace=True, errors='ignore')

# preprocessing
clean_tweets_df['no_emojis'] = clean_tweets_df['raw_content'].apply(
    remove_emoji)
clean_tweets_df['cleaned_text'] = clean_tweets_df['no_emojis'].apply(
    twitter_text_cleaning)
clean_tweets_df['cleaned_text_wo_punctuation'] = clean_tweets_df[
    'cleaned_text'].apply(
    remove_punc)

clean_tweets_df['cleaned_text_wo_punctuation_and_stopwords'] = clean_tweets_df[
    'cleaned_text_wo_punctuation'].apply(
    lambda x: remove_stopwords_and_tokenize(x, stop_words))

clean_tweets_df['preprocessed_text'] = clean_tweets_df[
    'cleaned_text_wo_punctuation_and_stopwords'].apply(" ".join)
# preprocessing

clean_tweets_df['word_count'] = clean_tweets_df[
    'cleaned_text_wo_punctuation_and_stopwords'].apply(len)
clean_tweets_df['word_count'] = clean_tweets_df['word_count'].astype(uint8)

clean_tweets_df['vocabulary'] = clean_tweets_df[
    'cleaned_text_wo_punctuation'].apply(
    lambda x: get_ngram_counts(x, stop_words))

# for idx, row in clean_tweets_df.iterrows():
#     df_ngrams = pd.DataFrame.from_dict(
#         clean_tweets_df['vocabulary'][0], orient='index', columns=['count'])
#     df_ngrams.reset_index(inplace=True)
#     df_ngrams.rename(columns={'index': 'ngram'}, inplace=True)

# for i in range(len(clean_tweets_df)):
#     ngram, count = zip(*clean_tweets_df.at[i, 'vocabulary'].items())
#     clean_tweets_df['ngram'] = ngram
#     clean_tweets_df['count'] = count

# for index, row in clean_tweets_df.iterrows():
#     ngram, count = zip(*row['vocabulary'].items())
#     clean_tweets_df.at[index, 'ngram'] = ngram
#     clean_tweets_df.at[index, 'count'] = count

clean_tweets_df[['ngram', 'count']] = clean_tweets_df.apply(
    lambda row: pd.Series(list(zip(*row['vocabulary'].items()))), axis=1)

clean_tweets_df = clean_tweets_df[clean_tweets_df['ngram'].notna()]
clean_tweets_df = clean_tweets_df[clean_tweets_df['count'].notna()]

clean_tweets_df['ngram'] = clean_tweets_df['ngram'].apply(list)
clean_tweets_df['count'] = clean_tweets_df['count'].apply(list)
clean_tweets_df['time_only'] = clean_tweets_df['date'].dt.time

clean_tweets_df['hour'] = clean_tweets_df['time_only'].apply(lambda x: x.hour)

# vectorizer = CountVectorizer(stop_words=stop_words)
# hashtags_vectors = vectorizer.fit_transform(
#     clean_tweets_df['hashtags'].apply(lambda x_array: ' '.join(x_array)
#     if x_array else ''))
#
# encoder = OneHotEncoder(dtype=uint8)
# hashtags_onehot = encoder.fit_transform(hashtags_vectors.toarray())
# hashtag_names = encoder.get_feature_names_out()
# hashtags_df = pd.DataFrame(hashtags_onehot.toarray(), columns=hashtag_names)
#
# hashtags_df: pd.DataFrame = clean_tweets_df['hashtags'].apply(
#     lambda x_array: pd.Series(x_array)).stack().str.get_dummies()
#
# # Change the column names
# hashtags_df.columns = ['hashtag_' + x_array for x_array in list(set(
#     hashtags_df.columns))]
#
# # Concatenate the hashtags_df with the original DataFrame
# processed_tweets: pd.DataFrame = pd.concat(
#     [clean_tweets_df, hashtags_df], axis=1)
# print(processed_tweets)
#
# hashtags_df.reset_index(inplace=True)
# hashtags_df.drop(['level_0', 'level_1'], axis=1, inplace=True)
# # Define the number of top hashtags you want to select
# n = 21
#
# # Get the frequency of each hashtag
# hashtag_counts = hashtags_df.sum().sort_values(ascending=False)
#
# # Select the top n hashtags
# top_hashtags = hashtag_counts.nlargest(n)
#
# hashtags_df = hashtags_df[
#     [x_array for x_array in hashtags_df.columns if 'hashtag_' + x_array in
#      top_hashtags.index]]


x_topics: np.ndarray = latent_dirichlet_allocation(
    clean_tweets_df, 'preprocessed_text', stop_words)

cluster_range = range(2, 10)
elbow_method(x_topics, cluster_range)
silhouette_scores(x_topics, cluster_range)

cluster_index: np.ndarray = kmeans_clustering(x_topics, 2)
visualize_clusters(x_topics, cluster_index)
insecurity_words: list[str] = json_file.get('words')
i_score: list[float16] = get_scores(
    " ".join(insecurity_words), clean_tweets_df['preprocessed_text'].to_list())

sns.displot(np.array(i_score))
plt.show()

clean_tweets_df['score'] = i_score
clean_tweets_df['insecurity'] = np.where(
    clean_tweets_df['score'] >= np.mean(np.array(i_score)) - np.std(
        np.array(i_score)), 1, 0)
clean_tweets_df['insecurity'] = clean_tweets_df['insecurity'].astype('uint8')
clean_tweets_df.drop(
    ["no_emojis", "cleaned_text", "cleaned_text_wo_punctuation",
     "cleaned_text_wo_punctuation_and_stopwords", "word_count"], axis=1,
    inplace=True)

tweets_df = clean_tweets_df.drop(
    ['date', 'reply_count', 'retweet_count', 'like_count',
     'quote_count', 'links', 'media', 'in_reply_to_user', 'mentioned_users',
     'view_count', 'user_username', 'user_created', 'user_followers_count',
     'user_friends_count', 'user_favourites_count', 'time_only', 'vocabulary',
     'hashtags', 'source_label', 'score',
     # 'location'
     ], axis=1)

# tweets_df['location'] = tweets_df['user_location'].where(
#     tweets_df['user_location'].notna(), tweets_df['place'])
# tweets_df['location'] = tweets_df['location'].where(
#     tweets_df['location'].notna(), tweets_df['coordinates'])
tweets_df = tweets_df.drop(['coordinates', 'place', 'user_location'], axis=1)
tweets_df['hour'] = tweets_df['hour'].astype(uint8)
tweets_df['insecurity'] = tweets_df['insecurity'].astype(uint8)


def update_target(
        list_of_dicts: list[dict], dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Updates tweet targets to 0 for any tweets in `tweets_df` that have
     a matching `id` in `additional_tweets`
    :param list_of_dicts: The list of tweets from other subject
    :type list_of_dicts: list[dict]
    :param dataframe: The tweets dataframe
    :type dataframe: pd.DataFrame
    :return: The fixed dataframe
    :rtype: pd.DataFrame
    """
    df_copy: pd.DataFrame = dataframe.copy()
    for tweet in list_of_dicts:
        tweet_id: int = tweet.get("id")
        if tweet_id is not None:
            mask = df_copy["id"] == tweet_id
            if mask.any():
                df_copy.loc[mask, "insecurity"] = 0
    return df_copy


print(tweets_df['insecurity'].value_counts())
updated_tweets: pd.DataFrame = update_target(additional_tweets, tweets_df)
tweets_df = updated_tweets.copy()
tweets_df = tweets_df.drop(['raw_content', 'id'], axis=1)
classified_tweets: bool = PersistenceManager.save_to_csv(
    data=tweets_df, filename='tweets_to_analyze.csv')
print(tweets_df['insecurity'].value_counts())
print(tweets_df.shape)
print("--------------------------------")

# tweets_df['location'] = tweets_df['location'].replace(
#     ['CUENCA, ATENAS DEL ECUADOR', 'Ecuador - Cuenca', 'Cuenca-Ecuador',
#      'CUENCA ECUADOR', 'Cuenca Ecuador', 'Cuenca - Ecuador ',
#      'Cuenca, Ecuador', 'Cuenca - Ecuador'], 'Cuenca')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Quito, Ecuador', 'Quito - Ecuador', 'Quito-Ecuador', 'QUITO',
#      'Quito Ecuador', 'Quito ', 'Quito , Ecuador', 'Quito -Ecuador',
#      'QUITO-ECUADOR', 'Quito, Ecuador ????????', 'Quito, Ecuador ',
#      'Ecuador-Quito',
#      'Quito D.M. - Ecuador ????????', 'Quito. Ecuador', 'Kito(Quito) Ecuador',
#      'QUITO, ECUADOR', 'Quito, Ec'], 'Quito')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Guayaquil, Ecuador', ' Guayaquil, Ecuador', 'Guayaquil, Ecuador ',
#      'Ecuador, Guayaquil', 'Guayaquil , Ecuador ', 'guayaquil', 'Guayaquil ',
#      'Guayaquil- Ecuador', 'Ecuador - Guayaquil', 'Guayaquil, ecuador',
#      'GUAYAQUIL-ECUADOR', 'Guayaquil - Ecuador', 'Guayaquil, Ecuador????????',
#      'Ecuador, Guayas, Guayaquil', 'Guayaquil-Ecuador', 'Guayaquil, ECUADOR',
#      'Guayaquil,Ecuador', 'Guayaquil de mis amores', 'Guayaquil City',
#      'guayaquil- ecuador', 'Guayaquil, Ecuador    Te sigue', 'Gye',
#      'Ecuador-Guayaquil', 'Guayaquil, EC', 'Guayaquil ???????? ', 'Gquil, Ecuador',
#      'Rep??blica de Guayaquil', 'Cedros y V??ctor Emilio Estrada'],
#     'Guayaquil')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['ECUADOR', 'Ecuador ', 'ecuador ', ' Ecuador', 'E C U A D O R ',
#      '        ECUADOR Pa??s Bendecido', 'ecuador', '#AllYouNeedIsEcuador',
#      '????????Ecuador????????', 'ECUADOR ', 'Ecuador,America del Sur'], 'Ecuador')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Loja, Ecuador', 'Loja-Ecuador', 'Loja-  Ecuador', 'Ecuador-Loja',
#      'LOJA ECUADOR', 'Loja - Ecuador'], 'Loja')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Macas - Morona Santiago', 'Macas - Ecuador'], 'Macas')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Manta, Manab??, Ecuador', 'Manta, Ecuador', 'Manta - Manab?? - Ecuador',
#      'manta, ecuador'], 'Manta')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Portoviejo - Ecuador', 'Portoviejo, Manab??', 'Portoviejo, Ecuador'],
#     'Portoviejo')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Ecuador-Duran', 'Duran, Ecuador'], 'Duran')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Samborondon, Ecuador', 'Edif. XIMA Of#315, Samborond??n'],
#     'Samborondon')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Sangolqu?? ', 'Sangolqui '], 'Sangolqui')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Machala, El Oro, Ecuador', 'Machala - Ecuador',
#      'Machala-El Oro-Ecuador',
#      'Machala Ecuador', 'Machala - El Oro - Ecuador', 'Machala, Ecuador'],
#     'Machala')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Salinas - Ecuador ', 'Salinas, Ecuador', 'Salinas,'
#                                                ' Sta Elena, Ecuador'],
#     'Salinas')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Milagro, Ecuador'], 'Milagro')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['??????New York City Love ??????????????', 'Nueva York', 'New York, NY'],
#     'New York')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Ecuador - Los R??os - Babahoyo', 'BABAHOYO - ECUADOR'], 'Babahoyo')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Ambato-Ecuador', 'Ambato, Ecuador'], 'Ambato')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['RIOBAMBA ', 'Riobamba, Ecuador', 'Riobamba-Ecuador',
#      'Riobamba - Ecuador'], 'Riobamba')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Latacunga, Ecuador', 'Latacunga - Ecuador'], 'Latacunga')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Ibarra - Ecuador ', 'Ibarra, Ecuador'], 'Ibarra')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Gal??pagos,  Ecuador ', 'Gal??pagos ', 'Galapagos-Ecuador'],
#     'Gal??pagos')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Manabi, Ecuador', 'Manab?? Ecuador '], 'Manab??')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Guayas, Ecuador'], 'Guayas')
# tweets_df['location'] = tweets_df['location'].replace(
#     ['Santo Domingo-Ecuador', 'Santo Domingo, Ecuador'], 'Santo Domingo')

bow, vect = text_to_bow(tweets_df, 'preprocessed_text')

iterate_models(bow, tweets_df, 'insecurity')

# data: dict = {'names': clean_tweets_df['user_username'].to_list(),
#               'insecurity_score': i_score}
# scores_df: pd.DataFrame = pd.DataFrame(data)
#
#
# # assign classes based on highest score
# def get_classes(scores: list[float]):
#     cluster: list[float] = []
#     another_topic: list[float] = []
#     for i in scores:
#         if i >= 0.73:
#             cluster.append(1)
#             another_topic.append(0)
#         else:
#             another_topic.append(1)
#             cluster.append(0)
#     return cluster, another_topic
#
#
# l1: list[float] = scores_df['insecurity_score'].to_list()
# insecurity, not_insecurity = get_classes(l1)
# data: dict = {'name': scores_df.names.to_list(), 'insecurity': insecurity,
#               'not_insecurity': not_insecurity}
# class_df: pd.DataFrame = pd.DataFrame(data)
# # grouping the tweets by username
# new_groups_df = class_df.groupby(['name']).sum()
# # add a new totals column
# new_groups_df['total'] = new_groups_df['insecurity'] + \
#                          new_groups_df['not_insecurity']
# # add a new totals row
# new_groups_df.loc["Total"] = new_groups_df.sum()
# print(new_groups_df)
#
# # K-Means clustering
# dataframe.a = StandardScaler().fit_transform(dataframe.a.values.reshape(
#     -1, 1))
# # if it's needed to standardize
#
# X = new_groups_df[['insecurity', 'not_insecurity']].values
# # Elbow Method
#
# within_cluster_sum_square = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300,
#                     random_state=0)
#     kmeans.fit(X)
#     within_cluster_sum_square.append(kmeans.inertia_)
# plt.plot(range(1, 11), within_cluster_sum_square)
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('wcss')
# plt.show()
#
# # fitting kmeans to dataset
# kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300,
#                 random_state=0)
# Y_kmeans = kmeans.fit_predict(X)
# # Visualising the clusters
# plt.scatter(X[Y_kmeans == 0, 0], X[Y_kmeans == 0, 1], s=70, c='violet',
#             label='Cluster 1')
# plt.scatter(X[Y_kmeans == 1, 0], X[Y_kmeans == 1, 1], s=70, c='cyan',
#             label='Cluster 2')
#
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#             s=70, c='black', label='Centroids')
# plt.title('Clusters of tweets in economic and social groups')
# plt.xlabel('economic tweets')
# plt.ylabel('social tweets')
# plt.legend()
# plt.show()

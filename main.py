"""
Main script for ML project
"""
import re
import string
from re import Pattern

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.utils import simple_preprocess
from matplotlib import pyplot as plt
from numpy import uint32, float16, uint8
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score

from engineering.persistence_manager import PersistenceManager, DataType
from engineering.snscrape_collection import decode_tweet_to_json, flatten, \
    str_to_datetime_values, nested_camel, get_nested_dict_structure, \
    combine_flattened
from schemas.filter import BetterFilter
from schemas.specification import TextSpecification, SenderSpecification

plt.rcParams.update({'figure.max_open_warning': 0})
json_file: dict[str, list[str]] = PersistenceManager.read_from_json()
print(json_file)
insecurity_words: list[str] = json_file.get('words')
users: list[str] = json_file.get('users')
better_filter: BetterFilter = BetterFilter()
# sender_spec: SenderSpecification = SenderSpecification('TDataScience')
# receiver_spec: ReceiverSpecification = ReceiverSpecification(
# 'JuanPabloCadena')

tweets_from_users: list[str] = []
for user in users:
    print(user)
    sender_spec: SenderSpecification = SenderSpecification(user)
    tweets_from_users.extend(better_filter.filter(
        sender_spec, limit=80, func=decode_tweet_to_json))

tweets_collected: list[dict] = []
for word in insecurity_words:
    print(word)
    if ' ' in word:
        print("space in:", word)
        continue
    text_spec: TextSpecification = TextSpecification(word)
    tweets_collected.extend(better_filter.filter(
        text_spec, limit=200, func=decode_tweet_to_json))

tweets_collected.extend(tweets_from_users)
# Save raw dataframe
raw_saved: bool = PersistenceManager.save_to_csv(
    tweets_collected, DataType.RAW.value, 'insecurity_raw_tweets')
if raw_saved:
    print('raw tweets saved!')

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
    'references/stop_words.json')
snscrape_columns: list[str] = stopwords_file.get('tweets')

# 'mm/dd/yyyy HH:MM:SS
# 'date', 'user_created'
clean_tweets_df.drop(snscrape_columns, axis=1, inplace=True, errors='ignore')
print(clean_tweets_df)

# Save processed dataframe
processed_saved: bool = PersistenceManager.save_to_csv(
    clean_tweets_df, DataType.PROCESSED.value,
    'insecurity_preprocessed_tweets')
if processed_saved:
    print('clean tweets saved!')

exclude_words: list[str] = stopwords_file.get('spanish')


def twitter_text_cleaning(text: str):
    """
    INPUT
    text - a text message that might have @mentions, "#" symbol or URLs

    OUTPUT
    text - the text without @mentions, "#" and URLs
    """
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'@[A-Za-zA-Z0-9]+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'@[A-Za-z]+', '', text)
    text = re.sub(r'@[-)]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'https?\/\/\S+', '', text)
    text = re.sub(r'http?\/\/\S+', '', text)
    text = re.sub(r'https?\/\/.*[\r\n]*', '', text)
    text = re.sub(r'^https?\/\/.*[\r\n]*', '', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    return text


def remove_emoji(text: str) -> str:
    emoji_pattern: Pattern[str] = re.compile(
        "["u"\U0001f600-\U0001f64f"  # emoticons
        u"\U0001f300-\U0001f5ff"  # symbols & pictographs
        u"\U0001f680-\U0001f6ff"  # transport & map symbols
        u"\U0001f1e0-\U0001f1ff"  # flags (iOS)
        u"\U0001f900-\U0001f9ff"  # Unicode 9.0 emojis
        u"\U0001f980-\U0001f9ff"  # Unicode 10.0 emojis
        u"\U0001fa80-\U0001faff"  # Unicode 11.0 emojis
        u"\U0001fbc0-\U0001fbc9"  # Unicode 12.0 emojis
        u"\U0001fcc0-\U0001fcc9"  # Unicode 13.0 emojis
        u"\U0001fcd0-\U0001fcd9"  # Unicode 14.0 emojis
        u"\U0001fdd0-\U0001fdd9"  # Unicode 15.0 emojis
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


clean_tweets_df['no_emojis'] = clean_tweets_df['raw_content'].apply(
    remove_emoji)
clean_tweets_df['cleaned_text'] = clean_tweets_df['no_emojis'].apply(
    twitter_text_cleaning)


def remove_punc(message):
    """
    INPUT
    message - a text message that might have punctuations in it

    OUTPUT
    message without punctuations
    """
    return ''.join(
        [char for char in message if char not in string.punctuation])


clean_tweets_df['cleaned_text_wo_punctuation'] = clean_tweets_df[
    'cleaned_text'].apply(
    remove_punc)
stop_words: list[str] = nltk.corpus.stopwords.words('spanish')
stop_words.extend(exclude_words)
stop_words = list(set(stop_words))


def stop_words_preprocess(text):
    """
    INPUT
    text - a text message that might have stop words

    OUTPUT
    list of words without stop words examples
    """
    return [w for w in simple_preprocess(text) if
            w not in stop_words and len(w) >= 3]


clean_tweets_df['cleaned_text_wo_punctuation_and_stopwords'] = clean_tweets_df[
    'cleaned_text_wo_punctuation'].apply(stop_words_preprocess)

clean_tweets_df['preprocessed_text'] = clean_tweets_df[
    'cleaned_text_wo_punctuation_and_stopwords'].apply(lambda x: " ".join(x))

clean_tweets_df['word_count'] = clean_tweets_df[
    'cleaned_text_wo_punctuation_and_stopwords'].apply(len)
clean_tweets_df['word_count'] = clean_tweets_df['word_count'].astype(uint8)
print(clean_tweets_df)


def word_counts_n_grams(tweet: str) -> dict:
    print("word_counts_n_grams")
    print(tweet)
    token_counts_matrix: CountVectorizer = CountVectorizer(
        stop_words=stop_words, ngram_range=(1, 3))
    print("test")
    try:
        doc_term_matrix = token_counts_matrix.fit_transform(tweet.split('\n'))
        word_counts = doc_term_matrix.toarray()
        print(word_counts)
    except Exception as e:
        doc_term_matrix = None
        print(e)
    print(type(doc_term_matrix))
    print(doc_term_matrix)
    vocabulary: dict
    try:
        vocabulary = token_counts_matrix.vocabulary_
        ngrams_count = dict(zip(vocabulary.keys(),
                                doc_term_matrix.sum(axis=0).tolist()[0]))
    except Exception as exc:
        print(exc)
        ngrams_count = {}
    print(ngrams_count)
    return ngrams_count


clean_tweets_df['vocabulary'] = clean_tweets_df[
    'cleaned_text_wo_punctuation'].apply(lambda x: word_counts_n_grams(x))
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


def tuple_to_list(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: list(x))
    return df


clean_tweets_df = clean_tweets_df[clean_tweets_df['ngram'].notna()]
clean_tweets_df = clean_tweets_df[clean_tweets_df['count'].notna()]

clean_tweets_df = tuple_to_list(clean_tweets_df, 'ngram')
clean_tweets_df = tuple_to_list(clean_tweets_df, 'count')
clean_tweets_df['time_only'] = clean_tweets_df['date'].dt.time


def extract_hour(timestamp):
    return timestamp.hour


clean_tweets_df['hour'] = clean_tweets_df['time_only'].apply(extract_hour)

# vectorizer = CountVectorizer(stop_words=stop_words)
# hashtags_vectors = vectorizer.fit_transform(
#     clean_tweets_df['hashtags'].apply(lambda x: ' '.join(x) if x else ''))
#
# encoder = OneHotEncoder(dtype=uint8)
# hashtags_onehot = encoder.fit_transform(hashtags_vectors.toarray())
# hashtag_names = encoder.get_feature_names_out()
# hashtags_df = pd.DataFrame(hashtags_onehot.toarray(), columns=hashtag_names)

hashtags_df: pd.DataFrame = clean_tweets_df['hashtags'].apply(
    lambda x: pd.Series(x)).stack().str.get_dummies()

# Change the column names
hashtags_df.columns = ['hashtag_' + x for x in list(set(hashtags_df.columns))]

# Concatenate the hashtags_df with the original DataFrame
processed_tweets: pd.DataFrame = pd.concat(
    [clean_tweets_df, hashtags_df], axis=1)
print(processed_tweets)


# Fixme: One Hot Encoding


# Don't need to reduce dimensions.

def latent_semantic_analysis(df: pd.DataFrame, column: str):
    tfidf_matrix: TfidfVectorizer = TfidfVectorizer(stop_words=stop_words)
    weighted_tfidf_matrix = tfidf_matrix.fit_transform(df[column])
    svd: TruncatedSVD = TruncatedSVD(n_components=100)
    svd.fit(weighted_tfidf_matrix)
    var: np.ndarray = svd.explained_variance_ratio_
    plt.plot(var)
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance ratio')
    plt.show()
    print(var)
    cumulative_explained_variance_ratio = np.cumsum(var)
    print(cumulative_explained_variance_ratio)
    # Find the number of components that retain the desired amount of variance
    n_components = np.argmax(cumulative_explained_variance_ratio >= 0.9) + 1
    print(n_components)
    reduced_matrix: np.ndarray = svd.fit_transform(weighted_tfidf_matrix)
    print(reduced_matrix)
    return reduced_matrix


reduced_x = latent_semantic_analysis(clean_tweets_df, 'preprocessed_text')


# LDA
def latent_dirichlet_allocation(dataframe: pd.DataFrame, column: str):
    token_counts_matrix: CountVectorizer = CountVectorizer(
        stop_words=stop_words, max_df=0.95, min_df=2)
    doc_term_matrix = token_counts_matrix.fit_transform(dataframe[column])
    lda_classifier: LatentDirichletAllocation = LatentDirichletAllocation(
        n_components=2, random_state=0)
    transformed_matrix = lda_classifier.fit_transform(doc_term_matrix)
    print(transformed_matrix)
    return transformed_matrix


x_topics = latent_dirichlet_allocation(clean_tweets_df, 'preprocessed_text')


def elbow_method(matrix, n_clusters_range):
    wcss = []
    for i in n_clusters_range:
        kmeans = KMeans(n_clusters=i, n_init=10)
        kmeans.fit(matrix)
        wcss.append(kmeans.inertia_)
    plt.plot(n_clusters_range, wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster sum of squares (WCSS)')
    plt.show()
    print(wcss)


def silhouette_scores(matrix, n_clusters_range):
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        labels = kmeans.fit_predict(matrix)
        silhouette_avg = silhouette_score(matrix, labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)


cluster_range = range(2, 10)
elbow_method(x_topics, cluster_range)
silhouette_scores(x_topics, cluster_range)


def kmeans_clustering(x_transformed, n_clusters):
    k_means = KMeans(n_clusters=n_clusters, n_init=10)
    y_pred = k_means.fit_predict(x_transformed)
    return y_pred


cluster_index = kmeans_clustering(x_topics, 2)
print(cluster_index)


def visualize_clusters(matrix, labels):
    # create scatter plot of data
    plt.scatter(matrix[:, 0], matrix[:, 1], c=labels, cmap='rainbow')
    plt.title("Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    legend_handles = [
        plt.Line2D([], [], color=plt.cm.rainbow(i / 2), label=f'Group {i}') for
        i in range(2)]
    plt.legend(handles=legend_handles)
    plt.show()

    # analyze cluster characteristics
    for i in range(np.unique(labels).shape[0]):
        cluster = matrix[labels == i]
        print("Cluster", i, ":")
        print("Number of samples:", cluster.shape[0])
        print("Mean:", np.mean(cluster, axis=0))
        print("Median:", np.median(cluster, axis=0))
        print("Standard deviation:", np.std(cluster, axis=0))
        print("\n")

    # identify outliers
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        outliers = matrix[labels == -1]
        print("Outliers:")
        print(outliers)


visualize_clusters(x_topics, cluster_index)


# df_ngrams = pd.DataFrame.from_dict(
#     clean_tweets_df['vocabulary'], orient='index')
# df_ngrams.reset_index(inplace=True)
# df_ngrams.columns = ['ngram', 'count']

# clean_tweets_df.tweet = clean_tweets_df.tweet.apply(furnished)
# Testing plot functions
# plot_count(clean_tweets_df, ['raw_content', 'retweet_count'], 'lang')
# plot_distribution(clean_tweets_df.date, 'red')
# boxplot_dist(clean_tweets_df, 'retweet_count', 'like_count')
# plot_scatter(clean_tweets_df, 'raw_content', 'date', 'lang')
# plot_heatmap(clean_tweets_df)


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def get_scores(group: list[str], tweets: list):
    scores: list[float] = []
    for tweet in tweets:
        s: float = jaccard_similarity(group, tweet)
        scores.append(s)
    return scores


i_score: list[float16] = get_scores(
    " ".join(insecurity_words), clean_tweets_df['preprocessed_text'].to_list())
print(i_score)
print(np.mean(np.array(i_score)), np.std(np.array(i_score)),
      np.min(np.array(i_score)), np.max(np.array(i_score)),
      np.median(np.array(i_score)))

sns.displot(np.array(i_score))
plt.show()

clean_tweets_df['score'] = i_score
clean_tweets_df['insecurity'] = np.where(
    clean_tweets_df['score'] >= np.mean(np.array(i_score)) - np.std(
        np.array(i_score)), 1, 0)
clean_tweets_df['insecurity'] = clean_tweets_df['insecurity'].astype('uint8')
print(clean_tweets_df['insecurity'].value_counts())

# Clustered DataFrame
data: dict = {'names': clean_tweets_df['user_username'].to_list(),
              'insecurity_score': i_score}
scores_df: pd.DataFrame = pd.DataFrame(data)


# assign classes based on highest score
def get_classes(scores: list[float]):
    cluster: list[float] = []
    another_topic: list[float] = []
    for i in scores:
        if i >= 0.73:
            cluster.append(1)
            another_topic.append(0)
        else:
            another_topic.append(1)
            cluster.append(0)
    return cluster, another_topic


l1: list[float] = scores_df['insecurity_score'].to_list()
insecurity, not_insecurity = get_classes(l1)
data: dict = {'name': scores_df.names.to_list(), 'insecurity': insecurity,
              'not_insecurity': not_insecurity}
class_df: pd.DataFrame = pd.DataFrame(data)
# grouping the tweets by username
new_groups_df = class_df.groupby(['name']).sum()
# add a new totals column
new_groups_df['total'] = new_groups_df['insecurity'] + \
                         new_groups_df['not_insecurity']
# add a new totals row
new_groups_df.loc["Total"] = new_groups_df.sum()
print(new_groups_df)

# K-Means clustering
# df.a = StandardScaler().fit_transform(df.a.values.reshape(-1, 1))
# if it's needed to standardize

X = new_groups_df[['insecurity', 'not_insecurity']].values
# Elbow Method

within_cluster_sum_square = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300,
                    random_state=0)
    kmeans.fit(X)
    within_cluster_sum_square.append(kmeans.inertia_)
plt.plot(range(1, 11), within_cluster_sum_square)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()

# fitting kmeans to dataset
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300,
                random_state=0)
Y_kmeans = kmeans.fit_predict(X)
# Visualising the clusters
plt.scatter(X[Y_kmeans == 0, 0], X[Y_kmeans == 0, 1], s=70, c='violet',
            label='Cluster 1')
plt.scatter(X[Y_kmeans == 1, 0], X[Y_kmeans == 1, 1], s=70, c='cyan',
            label='Cluster 2')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=70,
            c='black', label='Centroids')
plt.title('Clusters of tweets in economic and social groups')
plt.xlabel('economic tweets')
plt.ylabel('social tweets')
plt.legend()
plt.show()

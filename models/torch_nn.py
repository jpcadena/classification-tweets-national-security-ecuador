"""
Neural Network Model using Pytorch
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim


# Fixme: Improve the Pytorch implementation.


class TextClassifier(nn.Module):
    """
    Text Classifier class that inherits from Pytorch Neural Network
     Module.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """

        :param x:
        :type x:
        :return:
        :rtype:
        """
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def prepare_data(dataframe: pd.DataFrame):
    """

    :param dataframe:
    :type dataframe: pd.DataFrame
    :return:
    :rtype:
    """
    dataframe['ngram'] = dataframe['ngram'].apply(','.join)
    vocab_size = len(dataframe['ngram'].unique())
    # find the max length of the lists in the count column
    max_len = dataframe['count'].map(len).max()

    # pad the lists with 0 to make them all have the same length
    dataframe['count'] = dataframe['count'].apply(
        lambda x: x + [0] * (max_len - len(x)))

    # convert the lists of counts into a numpy array
    count_np = np.array(dataframe['count'].tolist(), dtype=np.float32)

    # convert the numpy array into a PyTorch tensor
    bow = torch.tensor(count_np)
    hour = torch.tensor(dataframe['hour'].values, dtype=torch.float32)
    target = torch.tensor(dataframe['insecurity'].values, dtype=torch.float32)
    return bow, hour, target, vocab_size


# df_exploded = tweets_df['ngram'].apply(pd.Series).merge(
#     tweets_df, right_index=True, left_index=True).drop(
#     ["ngram"], axis=1).melt(
#     id_vars=['count', 'hour', 'insecurity'], value_name="ngram").drop(
#     "variable", axis=1).drop_duplicates()


# df_exploded = tweets_df.apply(lambda x_array: pd.Series(
#     [(x_array['ngram'], x_array['count']) for i in range(len(
#         x_array['ngram']))]), axis=1).stack().reset_index(level=1, drop=True)
# df_exploded = df_exploded.to_frame(name='ngram_count').reset_index()
# dataframe['ngram'] = dataframe['ngram'].apply(
#     lambda x_array: ','.join(x_array))

def train(
        dataframe: pd.DataFrame, epochs, learning_rate, hidden_dim,
        embedding_dim):
    """

    :param dataframe:
    :type dataframe: pd.DataFrame
    :param epochs:
    :type epochs:
    :param learning_rate:
    :type learning_rate:
    :param hidden_dim:
    :type hidden_dim:
    :param embedding_dim:
    :type embedding_dim:
    :return:
    :rtype:
    """
    bow, hour, target, vocab_size = prepare_data(dataframe)
    model = TextClassifier(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(bow)
        loss = criterion(output.view(-1), target)
        # Backward pass
        loss.backward()
        optimizer.step()
        # Print the loss
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')


EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_DIM = 100
EMBEDDING_DIM = 100
# train(dataframe, epochs, learning_rate, hidden_dim, embedding_dim)

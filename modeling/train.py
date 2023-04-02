"""
This script contains a function to split the data into training and
 testing sets for a machine learning model.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


def training(
        bow: csr_matrix, dataframe: pd.DataFrame,
        target_column: str = "insecurity"
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets for a machine
     learning model. The function takes a bag-of-words representation
      of the text data and a Pandas DataFrame containing the target
       variable. The target variable is specified by the user as a
        column name in the DataFrame. The function uses the
         train_test_split function from scikit-learn to split the data
          into training and testing sets with a test size of 20%.
           The function returns the training and testing sets for both
            the input features and the target variable as NumPy arrays
    :param bow: A sparse matrix of the bag-of-words representation of
     the text data
    :type bow: csr_matrix
    :param dataframe: A Pandas DataFrame containing the target variable
    :type dataframe: pd.DataFrame
    :param target_column: The name of the target variable column in the
     DataFrame
    :type target_column: str
    :return: A tuple containing the training and testing sets for both
     the input features and the target variable
    :rtype: tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]
    """
    x_array: csr_matrix = bow
    y_array: np.ndarray = dataframe[target_column].values
    x_train, x_test, y_train, y_test = train_test_split(
        x_array, y_array, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

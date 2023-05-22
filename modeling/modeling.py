"""
Model prediction script
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from modeling.train import training


def predict_model(
    bow: csr_matrix,
    dataframe: pd.DataFrame,
    ml_model,
    target_column: str = "insecurity",
    boost: bool = False,
) -> tuple:
    """
    Predicts the target variable values using the provided model and
     returns the predicted and actual values.
    :param bow: The bag-of-words matrix representing the text data
    :type bow: csr_matrix
    :param dataframe: The pandas dataframe containing the text data and
     the target variable
    :type dataframe: pd.DataFrame
    :param ml_model: The machine learning model to use for prediction
    :type ml_model: Any
    :param target_column: The name of the target variable column in the
     dataframe
    :type target_column: str
    :param boost: Whether to boost the model training by converting
     data to float32
    :type boost: bool
    :return: A tuple of predicted and actual values for the target
     variable
    :rtype: tuple
    """
    x_train, x_test, y_train, y_test = training(bow, dataframe, target_column)
    if boost:
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        y_train = y_train.astype("float32")
        y_test = y_test.astype("float32")
    ml_model.fit(x_train, y_train)
    y_pred: np.ndarray = ml_model.predict(x_test)
    return y_pred, y_test

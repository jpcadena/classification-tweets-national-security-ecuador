"""
Neural Network using TensorFlow script
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, LSTM, GRU
from keras.models import Sequential
from scipy.sparse import csr_matrix

from modeling.evaluation import evaluate_model
from modeling.train import training

print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices("GPU"))
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)
session = tf.compat.v1.Session()
K.set_session(session)
tf.config.experimental.set_visible_devices(
    [tf.config.experimental.list_physical_devices("GPU")[0]], "GPU"
)


def reshape_array(matrix: csr_matrix) -> np.ndarray:
    """
    Reshape a csr_matrix to a 3D np.ndarray with shape
     (n_samples, 1, n_features)
    :param matrix: The matrix to reshape
    :type matrix: csr_matrix
    :return: The reshaped matrix
    :rtype: np.ndarray
    """
    array: np.ndarray[(int, int), np.float64] = matrix.toarray()
    return np.reshape(array, (array.shape[0], 1, array.shape[1]))


def train_nn(
    bow: csr_matrix, dataframe: pd.DataFrame, target_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train a neural network ml_model using the Bag of Words matrix and a
     pandas DataFrame
    :param bow: The Bag of Words matrix
    :type bow: csr_matrix
    :param dataframe: The pandas DataFrame with the target column
    :type dataframe: pd.DataFrame
    :param target_column: The target column name
    :type target_column: str
    :return: A tuple with the training and test data for both input and
     output
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    dataframe = dataframe.drop(["ngram", "count"], axis=1)
    x_train, x_test, y_train, y_test = training(bow, dataframe, target_column)
    x_train = reshape_array(x_train)
    x_test = reshape_array(x_test)
    return x_train, x_test, y_train, y_test


def predict_nn(
    bow: csr_matrix, dataframe: pd.DataFrame, target_column: str, layer: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the output for the neural network using the Bag of Words
     matrix and a pandas DataFrame
    :param bow: The Bag of Words matrix
    :type bow: csr_matrix
    :param dataframe: The pandas DataFrame with the target column
    :type dataframe: pd.DataFrame
    :param target_column: The target column name
    :type target_column: str
    :param layer: The type of layer to use (LSTM or GRU)
    :type layer: str
    :return: A tuple with the predicted output and the true output
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    sequential: Sequential = Sequential()
    x_train, x_test, y_train, y_test = train_nn(bow, dataframe, target_column)
    if layer == "LSTM":
        sequential.add(
            LSTM(100, input_shape=(1, x_train.shape[2]), return_sequences=False)
        )
    elif layer == "GRU":
        sequential.add(
            GRU(100, input_shape=(1, x_train.shape[2]), return_sequences=False)
        )
    sequential.add(Dense(1, activation="sigmoid"))
    sequential.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    sequential.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
    y_pred: np.ndarray = sequential.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int).flatten()
    return y_pred, y_test


def model_nn(
    bow: csr_matrix, dataframe: pd.DataFrame, target_column: str, layer: str
) -> None:
    """
    Model Neural Network
    :param bow: The Bag of Words matrix
    :type bow: csr_matrix
    :param dataframe: The pandas DataFrame with the target column
    :type dataframe: pd.DataFrame
    :param target_column: The target column name
    :type target_column: str
    :param layer: The type of layer to use (LSTM or GRU)
    :type layer: str
    :return: None
    :rtype: NoneType
    """
    y_pred_lstm, y_test = predict_nn(bow, dataframe, target_column, layer)
    conf_matrix: np.ndarray = evaluate_model(y_pred_lstm, y_test)
    print(conf_matrix)

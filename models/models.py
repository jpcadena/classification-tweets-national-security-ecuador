"""
Models iteration script
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from engineering.visualization import plot_confusion_matrix
from modeling.evaluation import evaluate_model
from modeling.modeling import predict_model


def iterate_models(
        bow: csr_matrix, dataframe: pd.DataFrame,
        target_column: str = "insecurity"
) -> None:
    """
    Iterates through a list of machine learning models and evaluates
     their performance on the input data
    :param bow: A sparse matrix containing the preprocessed input data
    :type bow: csr_matrix
    :param dataframe: A pandas DataFrame containing the input data
    :type dataframe: pd.DataFrame
    :param target_column: The name of the target column in the input
     data
    :type target_column: str
    :return: None
    :rtype: NoneType
    """
    models: list = [LogisticRegression(), SVC(), RandomForestClassifier(),
                    MultinomialNB(), DecisionTreeClassifier(),
                    KNeighborsClassifier(),
                    AdaBoostClassifier(),
                    XGBClassifier(tree_method="gpu_hist", gpu_id=0),
                    CatBoostClassifier(task_type="GPU", devices="0"),
                    LGBMClassifier(
                        device="gpu", gpu_platform_id=0, gpu_device_id=0)]
    model_names: list[str] = []
    boost_models: list[bool] = []
    for model in models:
        if isinstance(
                model, (XGBClassifier, CatBoostClassifier, LGBMClassifier)):
            model_names.append(model.__class__.__name__)
            boost_models.append(True)
        else:
            model_names.append(type(model).__name__)
            boost_models.append(False)
    for model, model_name, boost in zip(models, model_names, boost_models):
        print("\n\n", model_name)
        y_pred, y_test = predict_model(
            bow, dataframe, model, target_column, boost)
        conf_matrix: np.ndarray = evaluate_model(y_pred, y_test)
        plot_confusion_matrix(
            conf_matrix, ["Insecurity", "Not insecurity"], model_name)

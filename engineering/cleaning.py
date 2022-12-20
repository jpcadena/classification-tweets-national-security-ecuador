"""
Data cleaning
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


def lof_observation(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function identifies outliers with LOF method
    :param dataframe: Dataframe containing data
    :type dataframe: pd.DataFrame
    :return:
    :rtype: pd.DataFrame
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num_cols = dataframe.select_dtypes(include=numerics)
    df_outlier = df_num_cols.astype("float64")
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    clf.fit_predict(df_outlier)
    df_scores = clf.negative_outlier_factor_
    scores_df: pd.DataFrame = pd.DataFrame(np.sort(df_scores))
    scores_df.plot(stacked=True, xlim=[0, 20], color='r',
                   title='Visualization of outliers according to the LOF '
                         'method', style='.-')
    plt.show()
    plt.savefig('reports/figures/outliers.png')
    th_val = np.sort(df_scores)[2]
    outliers = df_scores > th_val
    dataframe = dataframe.drop(df_outlier[~outliers].index)
    print(dataframe.shape)
    return dataframe


def clear_outliers(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This function remove the outliers from specific column dataframe
    :param dataframe: Dataframe containing data
    :type dataframe: pd.DataFrame
    :param column: Column name
    :type column: str
    :return:
    :rtype: pd.DataFrame
    """
    first_quartile: float = dataframe[column].quantile(0.25)
    third_quartile: float = dataframe[column].quantile(0.75)
    iqr: float = third_quartile - first_quartile
    lower: float = first_quartile - 1.5 * iqr
    upper: float = third_quartile + 1.5 * iqr
    print("Age - Lower score: ", lower, "and upper score: ", upper)
    df_outlier = dataframe[column][(dataframe[column] > upper)]
    print(df_outlier)
    return dataframe

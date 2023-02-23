"""
Visualization script
"""
import itertools
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

RE_PATTERN: str = "([a-z])([A-Z])"
RE_REPL: str = r"\g<1> \g<2>"
PALETTE: str = 'pastel'
FIG_SIZE: tuple[int] = (15, 8)
colors: list[str] = ['lightskyblue', 'coral', 'palegreen']
FONT_SIZE: int = 15


# TODO: Add WORDCLOUD plot.


def plot_count(dataframe: pd.DataFrame, variables, hue: str) -> None:
    """
    This method plots the counts of observations from the given variables
    :param dataframe: dataframe containing tweets info
    :type dataframe: pd.DataFrame
    :param variables: list of columns to plot
    :type variables: list
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    plt.suptitle('Count-plot for Discrete variables')
    plot_iterator: int = 1
    for i in variables:
        plt.subplot(1, 3, plot_iterator)
        sns.countplot(x=dataframe[i], hue=dataframe[hue], palette=PALETTE)
        label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=i)
        plt.xlabel(label, fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plot_iterator += 1
        plt.savefig(f'reports/figures/discrete_{i}.png')
        plt.show()


def plot_distribution(df_column: pd.Series, color: str) -> None:
    """
    This method plots the distribution of the given quantitative
     continuous variable
    :param df_column: Single column
    :type df_column: pd.Series
    :param color: color for the distribution
    :type color: str
    :return: None
    :rtype: NoneType
    """
    label: str = re.sub(
        pattern=RE_PATTERN, repl=RE_REPL, string=str(df_column.name))
    sns.displot(x=df_column, kde=True, color=color, height=8, aspect=1.875)
    plt.title('Distribution Plot for ' + label)
    plt.xlabel(label, fontsize=FONT_SIZE)
    plt.ylabel('Frequency', fontsize=FONT_SIZE)
    plt.savefig('reports/figures/' + str(df_column.name) + '.png')
    plt.show()


def boxplot_dist(
        dataframe: pd.DataFrame, first_variable: str, second_variable: str
) -> None:
    """
    This method plots the distribution of the first variable data
    in regard to the second variable data in a boxplot
    :param dataframe: data to use for plot
    :type dataframe: pd.DataFrame
    :param first_variable: first variable to plot
    :type first_variable: str
    :param second_variable: second variable to plot
    :type second_variable: str
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    x_label: str = re.sub(
        pattern=RE_PATTERN, repl=RE_REPL, string=first_variable)
    y_label: str = re.sub(
        pattern=RE_PATTERN, repl=RE_REPL, string=second_variable)
    sns.boxplot(x=first_variable, y=second_variable, data=dataframe,
                palette=PALETTE)
    plt.title(x_label + ' in regards to ' + y_label, fontsize=FONT_SIZE)
    plt.xlabel(x_label, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.savefig(
        f'reports/figures/discrete_{first_variable}_{second_variable}.png')
    plt.show()


def plot_scatter(dataframe: pd.DataFrame, x_array: str, y_array: str, hue: str
                 ) -> None:
    """
    This method plots the relationship between x and y for hue subset
    :param dataframe: dataframe containing tweets
    :type dataframe: pd.DataFrame
    :param x_array: x-axis column name from dataframe
    :type x_array: str
    :param y_array: y-axis column name from dataframe
    :type y_array: str
    :param hue: grouping variable to filter plot
    :type hue: str
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    sns.scatterplot(x=x_array, data=dataframe, y=y_array, hue=hue,
                    palette=PALETTE)
    label: str = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=y_array)
    plt.title(f'{x_array} Wise {label} Distribution')
    print(dataframe[[x_array, y_array]].corr())
    plt.savefig(f'reports/figures/{x_array}_{y_array}_{hue}.png')
    plt.show()


def plot_heatmap(dataframe: pd.DataFrame) -> None:
    """
    Plot heatmap to analyze correlation between features
    :param dataframe: dataframe containing tweets
    :type dataframe: pd.DataFrame
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    sns.heatmap(data=dataframe.corr(), annot=True, cmap="RdYlGn")
    plt.title('Heatmap showing correlations among columns', fontsize=FONT_SIZE)
    plt.savefig('reports/figures/correlations_heatmap.png')
    plt.show()


def elbow_method(matrix: np.ndarray, n_clusters_range: range) -> None:
    """
    Perform elbow method for KMeans clustering to determine optimal
     number of clusters.
    :param matrix: The feature matrix of the data
    :type matrix: np.ndarray
    :param n_clusters_range: The range of number of clusters to test
    :type n_clusters_range: range
    :return: None
    :rtype: NoneType
    """
    wcss: list[float] = []
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


def visualize_clusters(matrix: np.ndarray, labels: np.ndarray) -> None:
    """
    Visualize clusters and display cluster characteristics.
    :param matrix: The feature matrix of the data.
    :type matrix: np.ndarray
    :param labels: The cluster labels assigned by the KMeans algorithm.
    :type labels: np.ndarray
    :return: None
    :rtype: NoneType
    """
    plt.scatter(matrix[:, 0], matrix[:, 1], c=labels, cmap='rainbow')
    plt.title("Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    legend_handles = [
        plt.Line2D([], [], color=plt.cm.rainbow(i / 2), label=f'Group {i}') for
        i in range(2)]
    plt.legend(handles=legend_handles)
    plt.show()
    for i in range(np.unique(labels).shape[0]):
        cluster = matrix[labels == i]
        print("Cluster", i, ":")
        print("Number of samples:", cluster.shape[0])
        print("Mean:", np.mean(cluster, axis=0))
        print("Median:", np.median(cluster, axis=0))
        print("Standard deviation:", np.std(cluster, axis=0))
        print("\n")
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        outliers = matrix[labels == -1]
        print("Outliers:")
        print(outliers)


def plot_confusion_matrix(
        conf_matrix: np.ndarray, classes: list[str], name: str,
        normalize: bool = False, title: str = 'Confusion matrix',
        cmap=plt.cm.Blues
) -> None:
    """
    This function plots the Confusion Matrix of the test and pred arrays
    :param conf_matrix:
    :type conf_matrix: np.ndarray
    :param classes: List of class names
    :type classes: list[str]
    :param name: Name of the model
    :type name: str
    :param normalize: Whether to normalize the confusion matrix or not.
     The default is False
    :type normalize: bool
    :param title: title for Confusion Matrix plot
    :type title: str
    :param cmap: Color map for the confusion matrix. The default is
     plt.cm.Blues
    :type cmap: plt.cm,
    :return: None
    :rtype: NoneType
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(
            axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(conf_matrix)
    plt.figure(figsize=FIG_SIZE)
    plt.rcParams.update({'font.size': 16})
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, color="blue")
    plt.yticks(tick_marks, classes, color="blue")
    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max(initial=0) / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]),
                                  range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="red" if conf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('reports/figures/' + name + '_confusion_matrix.png')
    plt.show()

"""
Visualization script
"""
import re
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

RE_PATTERN: str = "([a-z])([A-Z])"
RE_REPL: str = "\g<1> \g<2>"
PALETTE: str = 'pastel'
FIG_SIZE: tuple[int] = (15, 8)
colors: list[str] = ['lightskyblue', 'coral', 'palegreen']
FONT_SIZE: int = 15


def plot_count(dataframe: pd.DataFrame, variables) -> None:
    """
    This method plots the counts of observations from the given variables
    :param dataframe:
    :type dataframe: pd.DataFrame
    :param variables:
    :type variables:
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    plt.suptitle('Count-plot for Discrete variables')
    plot_iterator: int = 1
    for i in variables:
        plt.subplot(1, 3, plot_iterator)
        sns.countplot(x=dataframe[i], hue=dataframe.Exited, palette=PALETTE)
        label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=i)
        plt.xlabel(label, fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plot_iterator += 1
        plt.show()
        plt.savefig(f'reports/figures/discrete_{i}.png')


def plot_distribution(df_column: pd.Series, color: str) -> None:
    """
    This method plots the distribution of the given quantitative
     continuous variable
    :param df_column: Single column
    :type df_column: pd.Series
    :param color:
    :type color:
    :return: None
    :rtype: NoneType
    """
    label = re.sub(pattern=RE_PATTERN, repl=RE_REPL,
                   string=str(df_column.name))
    sns.displot(x=df_column, kde=True, color=color, height=8, aspect=1.875)
    plt.title('Distribution Plot for ' + label)
    plt.xlabel(label, fontsize=FONT_SIZE)
    plt.ylabel('Frequency', fontsize=FONT_SIZE)
    plt.show()
    plt.savefig('reports/figures/' + str(df_column.name) + '.png')


def boxplot_dist(
        dataframe: pd.DataFrame, first_variable: str, second_variable: str
) -> None:
    """
    This method plots the distribution of the first variable data
    in regard to the second variable data in a boxplot
    :param dataframe: data to use for plot
    :type dataframe: pd.DataFrame
    :param first_variable: first variable
    :type first_variable: str
    :param second_variable: second variable
    :type second_variable: str
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    x_label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=first_variable)
    y_label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=second_variable)
    sns.boxplot(x=first_variable, y=second_variable, data=dataframe,
                palette=PALETTE)
    plt.title(x_label + ' in regards to ' + y_label, fontsize=FONT_SIZE)
    plt.xlabel(x_label, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.show()
    plt.savefig(
        f'reports/figures/discrete_{first_variable}_{second_variable}.png')


def plot_scatter(dataframe: pd.DataFrame, x: str, y: str, hue: str) -> None:
    """
    This method plots the relationship between x and y for hue subset
    :param dataframe:
    :type dataframe: pd.DataFrame
    :param x:
    :type x:
    :param y:
    :type y:
    :param hue:
    :type hue:
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    sns.scatterplot(x=x, data=dataframe, y=y, hue=hue,
                    palette=PALETTE)
    label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=y)
    plt.title(f'{x} Wise {label} Distribution')
    plt.show()
    print(dataframe[[x, y]].corr())
    plt.savefig(f'reports/figures/{x}_{y}_{hue}.png')


def plot_heatmap(dataframe: pd.DataFrame) -> None:
    """
    Plot heatmap to analyze correlation between features
    :param dataframe:
    :type dataframe: pd.DataFrame
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    sns.heatmap(data=dataframe.corr(), annot=True, cmap="RdYlGn")
    plt.title('Heatmap showing correlations among columns', fontsize=FONT_SIZE)
    plt.show()
    plt.savefig('reports/figures/correlations_heatmap.png')

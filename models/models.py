"""
Models script
"""
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scikitplot.metrics import plot_roc
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, \
    roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from engineering.visualization import FIG_SIZE

# TODO: Check data type of parameters and fill documentation of the functions.

def analyze_models(x_train, y_train, x_test, y_test) -> pd.DataFrame:
    """
    Function to analyze some Classification models from Scikit-Learn
    :param x_train:
    :type x_train:
    :param y_train:
    :type y_train:
    :param x_test:
    :type x_test:
    :param y_test:
    :type y_test:
    :return: results ordered by accuracy score
    :rtype: pd.DataFrame
    """
    models = [('LOGR', LogisticRegression()), ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('RF', RandomForestClassifier()),
              ('SVC', SVC()), ('GBM', GradientBoostingClassifier()),
              ('XGBoost', XGBClassifier()), ('LightGBM', LGBMClassifier()),
              ('CatBoost', CatBoostClassifier()),
              ('ABoost', AdaBoostClassifier())]
    models_df: pd.DataFrame = pd.DataFrame(
        columns=["model", "accuracy_score", "scale_method",
                 "0_precision", "0_recall", "1_precision",
                 "1_recall"])
    index: int = 0
    for name, model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, digits=2,
                                             output_dict=True)
        zero_report = class_report['0']
        one_report = class_report['1']
        models_df.at[index, ['model', 'accuracy_score', 'scale_method',
                             "0_precision", "0_recall", "1_precision",
                             "1_recall"]] = [name, score, "NA",
                                             zero_report['precision'],
                                             zero_report['recall'],
                                             one_report['precision'],
                                             one_report['recall']]
        index += 1
    models_df = models_df.sort_values("accuracy_score", ascending=False)
    return models_df


def plot_confusion_matrix(
        cm, classes, name, normalize=False, title='Confusion matrix',
        cmap=plt.cm.Blues) -> None:
    """
    This function plots the Confusion Matrix of the test and pred arrays
    :param cm:
    :type cm:
    :param classes:
    :type classes:
    :param name:
    :type name:
    :param normalize:
    :type normalize:
    :param title:
    :type title:
    :param cmap:
    :type cmap:
    :return: None
    :rtype: NoneType
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.figure(figsize=FIG_SIZE)
    plt.rcParams.update({'font.size': 16})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, color="blue")
    plt.yticks(tick_marks, classes, color="blue")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('reports/figures/' + name + '_confusion_matrix.png')


def get_model_results(model, x, y, y2, x2, classes: list[str],
                      title: str = ' Confusion Matrix') -> None:
    """
    Get Confusion matrix results and ROC as figures
    :param model:
    :type model:
    :param x:
    :type x:
    :param y:
    :type y:
    :param y2:
    :type y2:
    :param x2:
    :type x2:
    :param classes: name of axis for confusion matrix
    :type classes: list[str]
    :param title: Title for confusion matrix
    :type title: str
    :return: None
    :rtype: NoneType
    """
    model.fit(x, y)
    model_name = type(model).__name__
    predictions = model.predict(x2)
    print(model_name)
    print(classification_report(y2, predictions))
    auc = roc_auc_score(y2, predictions)
    print(auc)
    # CONFUSION MATRIX
    cfm = confusion_matrix(y2, y_pred=predictions)
    confusion_matrix_title = str(model_name) + title
    name = str(model_name)
    plot_confusion_matrix(
        cfm, classes=classes, name=name, title=confusion_matrix_title)
    true_negative, false_positive, false_negative, true_positive = cfm.ravel()
    print("True Negatives: ", true_negative)
    print("False Positives: ", false_positive)
    print("False Negatives: ", false_negative)
    print("True Positives: ", true_positive)
    # ROC CURVE
    y_pred_proba = model.predict_proba(x2)
    plot_roc(y2, y_pred_proba, figsize=FIG_SIZE)
    plt.show()
    plt.savefig('reports/figures/' + str(model_name) + '_roc_curves.png')

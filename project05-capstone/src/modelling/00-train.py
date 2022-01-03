import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

import argparse
import os
import logging
import sys

work_dir = os.getcwd()
sys.path.append(work_dir)

from src.utils.config import read_config

logging.basicConfig(
    format="%(filename)s %(asctime)s %(levelname)s Line no: %(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
    level=logging.INFO,
)

log = logging.getLogger(__name__)


def evaluate_performance(
    y_true: pd.Series,
    y_pred: pd.Series,
    print_results: bool = True,
    evaluation_type: str = "test",
):
    """
    Evaluate classification model performance.

    :param y_true: actual class
    :type y_true: pd.Series
    :param y_pred: predicted class
    :type y_pred: pd.Series
    :param print_results: if True, show evaluation metrics in the console, defaults to True
    :type print_results: bool, optional
    :param evaluation_type: dataset used to generate y_pred, e.g., train, validation, or test
    :type evaluation_type: str
    :return: tuple of three elements (accuracy, F1 score, and AUC)
    :rtype: tuple
    """
    assert evaluation_type in [
        "train",
        "validation",
        "test",
    ], f"Invalid {evaluation_type}! It must be either 'train', 'validation', or 'test'."

    accuracy = accuracy_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    if print_results:
        log.info(f"{evaluation_type} set - Accuracy : {accuracy:.3%}")
        log.info(f"{evaluation_type} set - F1-score : {f_score:.3%}")
        log.info(f"{evaluation_type} set - AUC: {auc:.3f}")
        log.info(classification_report(y_true, y_pred))

    return accuracy, f_score, auc


def kfold_cv(clf, X_train, y_train, k: int = 10, random_state: int = 121):
    kfold = KFold(random_state=random_state, shuffle=True, n_splits=k)

    cv_accuracy = np.zeros(shape=k)
    cv_f1 = np.zeros(shape=k)
    cv_auc = np.zeros(shape=k)

    i = 0
    for train_index, val_index in kfold.split(X_train):
        X_tr = X_train.loc[train_index]
        y_tr = y_train.loc[train_index]
        X_val = X_train.loc[val_index]
        y_val = y_train.loc[val_index]

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        accuracy, f_score, auc = evaluate_performance(
            y_val, y_pred, print_results=False
        )
        log.info(
            f"Iteration {i+1}: Accuracy={accuracy:.3%} | F1-score={f_score:.3%} | AUC={auc:.3f}"
        )

        cv_accuracy[i] = accuracy
        cv_f1[i] = f_score
        cv_auc[i] = auc

        i += 1

    log.info("Cross-validation results")
    log.info("========================")
    log.info(f"CV Accuracy: {np.mean(cv_accuracy):.3%} +- {np.std(cv_accuracy):.3%}")
    log.info(f"CV F1-score: {np.mean(cv_f1):.3%} +- {np.std(cv_f1):.3%}")
    log.info(f"CV AUC: {np.mean(cv_auc):.3f} +- {np.std(cv_auc):.3f}")


if __name__ == "__main__":
    log.info(
        "TO DO - add CLI args to get the hyperparameters if we want to do hp tuning (otherwise, read model config)"
    )

    ## read data
    config = read_config("feature-engineering.yml")
    storage_type = "local_storage"

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for dataset_type in ["train_set", "test_set"]:
        input_path = config[storage_type]["feature_set"][dataset_type]["path"]
        input_separator = config[storage_type]["feature_set"][dataset_type]["separator"]
        if "train" in dataset_type:
            neighbour_count = config["strategy"]["oversampling"]["neighbour_count"]
            input_path = input_path.replace(
                "df_train.csv", f"df_train-smote-{neighbour_count}.csv"
            )
            df_train = pd.read_csv(input_path, sep=input_separator)
        elif "test" in dataset_type:
            df_test = pd.read_csv(input_path, sep=input_separator)

    target_class = "order"
    y_train = df_train[target_class].copy()
    X_train = df_train.drop(labels=[target_class], axis=1)

    y_test = df_test[target_class].copy()
    X_test = df_test.drop(labels=[target_class], axis=1)

    random_state = 121
    ## test model training
    model_clf = RandomForestClassifier(random_state=random_state)

    ## Perform k-fold cross-validation
    kfold_iteration = 5
    kfold_cv(model_clf, X_train, y_train, k=kfold_iteration, random_state=random_state)

    ## Train with the whole training set
    model_clf.fit(X_train, y_train)

    y_pred = model_clf.predict(X_test)
    accuracy, f_score, auc = evaluate_performance(y_test, y_pred)

    try:
        df_model_coef = pd.DataFrame(
            data={
                "feature_name": X_train.columns,
                "coef": model_clf.coef_[0],  ## logistic regression
            }
        ).sort_values(by="coef", ascending=False)
    except:
        df_model_coef = pd.DataFrame(
            data={
                "feature_name": X_train.columns,
                "coef": model_clf.feature_importances_,  ## tree-based model
            }
        ).sort_values(by="coef", ascending=False)

    df_model_coef["coef_abs"] = np.abs(df_model_coef["coef"])

    log.info(model_clf.get_params())
    breakpoint()

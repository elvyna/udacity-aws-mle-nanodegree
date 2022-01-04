import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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


def check_outliers(series: pd.Series):
    """
    Return outlier flag and the number of outliers in the series.
    Outliers: if < Q1 - 1.5*IQR or > Q3 + 1.5*IQR

    :param series: series of numeric values to be analysed
    :type series: pd.Series
    :return: outlier flag (pd.Series) and number of outliers in that series
    :rtype: tuple
    """
    q1, q3 = np.percentile(series, q=[25, 75])
    iqr = q3 - q1
    lower_threshold = q1 - (1.5 * iqr)
    upper_threshold = q3 + (1.5 * iqr)
    conditions = [(series < lower_threshold) | (series > upper_threshold)]
    choice = [1]
    outlier_flag_list = np.select(conditions, choice, default=0)

    return outlier_flag_list, sum(outlier_flag_list)


if __name__ == "__main__":
    log.info(
        "Reference: \n 1) notebook/00-data-understanding.ipynb \n 2) notebook/02-model-training-iter1.ipynb"
    )

    ## read data
    config = read_config("data-preparation.yml")
    storage_type = "local_storage"
    input_path = config[storage_type]["reformat"]["path"]
    input_separator = config[storage_type]["reformat"]["separator"]

    df_source = pd.read_csv(input_path, sep=input_separator)
    log.info(f"Dataset preview \n{df_source.iloc[0]}")

    ## identify outliers
    numeric_column_list = df_source._get_numeric_data().columns
    excluded_column_list = ["address", "order"]
    selected_column_list = numeric_column_list.drop(labels=excluded_column_list)

    log.info("Number of outliers")
    log.info("------------------------")
    df_outliers = pd.DataFrame(
        data={"colnames": [], "outliers_count": [], "outliers_pct": []}
    )

    for col in selected_column_list:
        outlier_flag, outlier_count = check_outliers(df_source[col])
        df_source[col + "_is_outlier"] = outlier_flag

        new_df = pd.DataFrame(
            data={
                "colnames": col,
                "outliers_count": outlier_count,
                "outliers_pct": outlier_count / df_source.shape[0] * 100,
            },
            index=[0],
        )
        df_outliers = pd.concat([df_outliers, new_df])

    log.info(f"Identified outliers \n {df_outliers}")

    ## split training and test set
    ## train test split (using original data)
    test_proportion = config["strategy"]["test_proportion"]
    random_state = config["strategy"]["random_state"]
    target_class = "order"
    features = df_source.columns[(df_source.columns != target_class)]

    X_train, X_test, y_train, y_test = train_test_split(
        df_source[features],
        df_source[target_class],
        test_size=test_proportion,
        random_state=random_state,
    )

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    ## remove outliers ONLY on training set
    remove_outliers = config["strategy"]["remove_outliers"]
    row_count_before = df_train.shape[0]
    log.info(f"Row count before outlier removal: {row_count_before:,}")
    if remove_outliers:
        ## remove rows that contain outlier
        mask_is_outlier_column = df_train.columns.str.endswith("_is_outlier")
        is_outlier_column_list = df_train.columns[mask_is_outlier_column]

        for col in is_outlier_column_list:
            mask_not_outlier = df_train[col] == 0
            df_train = df_train[mask_not_outlier].copy().reset_index(drop=True)

        row_count_after = df_train.shape[0]
        log.info(
            f"Row count after outlier removal: {row_count_after:,} ({row_count_after / row_count_before:,.3%} of the original rows)"
        )

        ## remove is_outlier columns
        mask_is_outlier_column = df_train.columns.str.endswith("_is_outlier")
        is_outlier_column_list = df_train.columns[mask_is_outlier_column]
        df_train.drop(labels=is_outlier_column_list, axis=1, inplace=True)

        mask_is_outlier_column = df_test.columns.str.endswith("_is_outlier")
        is_outlier_column_list = df_test.columns[mask_is_outlier_column]
        df_test.drop(labels=is_outlier_column_list, axis=1, inplace=True)

    # save results
    for dataset_type, df in zip(["train_set", "test_set"], [df_train, df_test]):
        output_path = config[storage_type][dataset_type]["path"]
        log.info(f"Saving {dataset_type} with dimension {df.shape} to {output_path}")
        df.to_csv(output_path, index=False)

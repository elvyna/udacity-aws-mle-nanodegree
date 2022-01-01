import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

## use one hot encoder for categorical values
from sklearn.preprocessing import OneHotEncoder

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

## create categorical values based on startHour
def determine_time_of_day(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Determine time of day based on hour of the day.
    Time of day: early morning, morning, afternoon, or evening.

    :param df: input dataframe to be processed
    :type df: pd.DataFrame
    :param column_name: column that stores hour of day
    :type column_name: str
    :return: input dataframe with additional time_of_day column
    :rtype: pd.DataFrame
    """
    condition_list = [
        ((df[column_name] >= 0) & (df[column_name] < 6)),
        ((df[column_name] >= 6) & (df[column_name] < 12)),
        ((df[column_name] >= 12) & (df[column_name] < 18)),
        ((df[column_name] >= 18) & (df[column_name] < 25)),
    ]

    choice_list = ["early_morning", "morning", "afternoon", "evening"]

    df["time_of_day"] = np.select(condition_list, choice_list, default="unknown")
    return df


if __name__ == "__main__":
    ## read data
    config = read_config("feature-engineering.yml")
    storage_type = "local_storage"

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for dataset_type in ["train_set", "test_set"]:
        input_path = config[storage_type]["input"][dataset_type]["path"]
        input_separator = config[storage_type]["input"][dataset_type]["separator"]
        if "train" in dataset_type:
            df_train = pd.read_csv(input_path, sep=input_separator)
        elif "test" in dataset_type:
            df_test = pd.read_csv(input_path, sep=input_separator)

    df_train = determine_time_of_day(df=df_train, column_name="startHour")
    df_test = determine_time_of_day(df=df_test, column_name="startHour")

    ## OHE
    selected_feature_list = ["availability", "address", "time_of_day", "onlineStatus"]
    for selected_feature in selected_feature_list:
        ohe = OneHotEncoder()
        feature_array = np.array(df_train[selected_feature])
        feature_encoded = ohe.fit_transform(
            np.reshape(feature_array, (-1, 1))
        ).toarray()
        ## test set
        feature_array_test = np.array(df_test[selected_feature])
        feature_encoded_test = ohe.transform(
            np.reshape(feature_array_test, (-1, 1))
        ).toarray()

        try:
            feature_encoded = pd.DataFrame(
                data=feature_encoded,
                columns=[
                    selected_feature + "_" + col.str.replace(" ", "_")
                    for col in ohe.categories_[0]
                ],
            )
            feature_encoded_test = pd.DataFrame(
                data=feature_encoded_test,
                columns=[
                    selected_feature + "_" + col.str.replace(" ", "_")
                    for col in ohe.categories_[0]
                ],
            )
        except:
            log.info(selected_feature)
            feature_encoded = pd.DataFrame(
                data=feature_encoded,
                columns=[
                    selected_feature + "_" + str(col).replace(" ", "_")
                    for col in ohe.categories_[0]
                ],
            )
            feature_encoded_test = pd.DataFrame(
                data=feature_encoded_test,
                columns=[
                    selected_feature + "_" + str(col).replace(" ", "_")
                    for col in ohe.categories_[0]
                ],
            )

        feature_encoded.drop(feature_encoded.columns[-1], axis=1, inplace=True)
        feature_encoded_test.drop(
            feature_encoded_test.columns[-1], axis=1, inplace=True
        )

        df_train = pd.concat([df_train, feature_encoded], axis=1)
        df_test = pd.concat([df_test, feature_encoded_test], axis=1)

        ## remove redundant features - we've done one hot encoding
        df_train.drop(labels=selected_feature, axis=1, inplace=True)
        df_test.drop(labels=selected_feature, axis=1, inplace=True)

    ## remove is_outlier columns
    mask_is_outlier_column = df_train.columns.str.endswith("_is_outlier")
    is_outlier_column_list = df_train.columns[mask_is_outlier_column]
    df_train.drop(labels=is_outlier_column_list, axis=1, inplace=True)
    df_test.drop(labels=is_outlier_column_list, axis=1, inplace=True)

    ## reformat
    categorical_feature_list = [
        "availability_-99",
        "availability_completely_not_determinable",
        "availability_completely_not_orderable",
        "availability_completely_orderable",
        "availability_mainly_not_determinable",
        "availability_mainly_not_orderable",
        "availability_mainly_orderable",
        "address_-99",
        "address_1",
        "address_2",
        "time_of_day_afternoon",
        "time_of_day_early_morning",
        "time_of_day_evening",
        "onlineStatus_-99",
        "onlineStatus_0",
    ]

    for col in categorical_feature_list:
        df_train[col] = df_train[col].astype(int)
        df_test[col] = df_test[col].astype(int)

    # save results
    for dataset_type, df in zip(["train_set", "test_set"], [df_train, df_test]):
        output_path = config[storage_type]["feature_set"][dataset_type]["path"]
        log.info(f"Saving {dataset_type} with dimension {df.shape} to {output_path}")
        df.to_csv(output_path, index=False)

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

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

if __name__ == "__main__":
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

    selected_feature_proportion = config["strategy"]["feature_selection"]["proportion"]

    selected_feature_count = int(
        np.round(selected_feature_proportion * df_train.shape[1])
    )
    random_state = config["strategy"]["random_state"]
    rfe = RFE(
        estimator=RandomForestClassifier(random_state=random_state),
        n_features_to_select=selected_feature_count,
    )

    rfe.fit(X_train, y_train)

    mask_rfe_features = rfe.get_support()
    log.info(f"{selected_feature_count} important features")
    log.info(X_train.columns[mask_rfe_features])

    rfe_column_list = (X_train.columns[mask_rfe_features]).tolist()
    X_train_rfe = X_train[rfe_column_list].copy()

    df_train_rfe = pd.concat([X_train_rfe, y_train], axis=1)
    log.info(df_train_rfe.shape)

    df_test_rfe = df_test[rfe_column_list + [target_class]].copy()
    log.info(df_test_rfe.shape)

    # save results
    for dataset_type, df in zip(["train_set", "test_set"], [df_train, df_test]):
        output_path = config[storage_type]["feature_selection"][dataset_type]["path"]
        log.info(f"Saving {dataset_type} with dimension {df.shape} to {output_path}")
        df.to_csv(output_path, index=False)

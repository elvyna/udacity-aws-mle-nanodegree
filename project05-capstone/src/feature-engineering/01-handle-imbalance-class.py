import pandas as pd
from imblearn.over_sampling import SMOTE

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
    input_train_path = os.path.join("dataset", "feature-engineering", "df_train.csv")
    config = read_config("feature-engineering.yml")
    storage_type = "local_storage"
    input_path = config[storage_type]["feature_set"]["train_set"]["path"]
    input_separator = config[storage_type]["feature_set"]["train_set"]["separator"]
    df_train = pd.read_csv(input_train_path, sep=input_separator)

    target_class = "order"
    y_train = df_train[target_class].copy()
    X_train = df_train.drop(labels=[target_class], axis=1)

    random_state = config["strategy"]["random_state"]
    neighbour_count = config["strategy"]["oversampling"]["neighbour_count"]
    smote = SMOTE(random_state=random_state, k_neighbors=neighbour_count)

    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    log.info(f"Training data before SMOTE: {X_train.shape[0]:,.0f} records")
    log.info(f"Training data after SMOTE: {X_train_smote.shape[0]:,.0f} records")

    df_train_smote = pd.concat([X_train_smote, y_train_smote], axis=1)

    file_name = f"df_train-smote-{neighbour_count}.csv"
    output_path = os.path.join("dataset", "feature-engineering", file_name)
    log.info(f"Saving dataset with dimension {df_train.shape} to {output_path}")
    df_train_smote.to_csv(output_path, index=False)

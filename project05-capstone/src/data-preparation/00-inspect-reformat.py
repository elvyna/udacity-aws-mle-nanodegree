import pandas as pd
import numpy as np
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
    log.info("Reference: notebook/00-data-understanding.ipynb")

    config = read_config("data-preparation.yml")

    ## read raw data
    storage_type = "local_storage"
    input_path = config[storage_type]["raw"]["path"]
    input_separator = config[storage_type]["raw"]["separator"]

    df_source = pd.read_csv(input_path, sep=input_separator)
    log.info(f"Dataset preview \n{df_source.iloc[0]}")

    for col in df_source.columns:
        log.info(
            f"Preview of {col} - there are {df_source[col].nunique():,} unique values."
        )
        log.info(sorted(df_source[col].unique()[-5:]))
        log.info("--------------------------------")

        ## replace missing value with a numeric value, e.g., -99
        mask = df_source[col] == "?"
        df_source.loc[mask, col] = -99

    ## reformat data types
    # then, convert to numeric
    numeric_column_list = [
        "startHour",
        "startWeekday",
        "duration",
        "cCount",
        "cMinPrice",
        "cMaxPrice",
        "cSumPrice",
        "bCount",
        "bMinPrice",
        "bMaxPrice",
        "bSumPrice",
        "bStep",
        "customerNo",
        "maxVal",
        "customerScore",
        "accountLifetime",
        "payments",
        "age",
        "address",
        "lastOrder",
    ]

    for col in numeric_column_list:
        df_source[col] = df_source[col].astype(float)

    for col in ["onlineStatus", "order"]:
        condition_list = [
            (df_source[col] == "y"),
            (df_source[col] == "n"),
        ]
        choice_list = [1, 0]
        df_source[col] = np.select(condition_list, choice_list, default=df_source[col])

    for col in ["order", "address"]:
        df_source[col] = df_source[col].astype(int)

    ## remove ID columns
    df_source.drop(labels=["sessionNo", "customerNo"], axis=1, inplace=True)

    ## save results
    output_path = config[storage_type]["reformat"]["path"]
    log.info(f"Saving dataset with dimension {df_source.shape} to {output_path}")
    df_source.to_csv(output_path, index=False)

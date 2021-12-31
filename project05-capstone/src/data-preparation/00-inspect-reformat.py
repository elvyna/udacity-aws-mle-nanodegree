import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(
    format="%(filename)s %(asctime)s %(levelname)s Line no: %(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
    level=logging.INFO,
)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    log.info("Reference: notebook/00-data-understanding.ipynb")

    ## read raw data
    input_train_path = os.path.join("dataset", "transact_train.txt")

    df_train = pd.read_csv(input_train_path, sep="|")
    log.info(f"Dataset preview \n{df_train.iloc[0]}")

    for col in df_train.columns:
        log.info(
            f"Preview of {col} - there are {df_train[col].nunique():,} unique values."
        )
        log.info(sorted(df_train[col].unique()[-5:]))
        log.info("--------------------------------")

        ## replace missing value with a numeric value, e.g., -99
        mask = df_train[col] == "?"
        df_train.loc[mask, col] = -99

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
        df_train[col] = df_train[col].astype(float)

    for col in ["onlineStatus", "order"]:
        condition_list = [
            (df_train[col] == "y"),
            (df_train[col] == "n"),
        ]
        choice_list = [1, 0]
        df_train[col] = np.select(condition_list, choice_list, default=df_train[col])

    for col in ["order", "address"]:
        df_train[col] = df_train[col].astype(int)

    ## save results
    output_path = os.path.join("dataset", "preprocessed", "transact_train_reformat.csv")
    log.info(f"Saving dataset with dimension {df_train.shape} to {output_path}")
    df_train.to_csv(output_path, index=False)

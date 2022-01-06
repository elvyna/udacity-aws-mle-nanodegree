import json
import logging
import sys
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
from io import StringIO

logging.basicConfig(
    format="%(filename)s %(asctime)s %(levelname)s Line no: %(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

log.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"
DEFAULT_COLUMN_NAME_LIST = [
    "sessionNo", "startHour", "startWeekday", "duration", "cCount", "cMinPrice", "cMaxPrice","cSumPrice",
    "bCount", "bMinPrice", "bMaxPrice", "bSumPrice", "bStep", "onlineStatus", "availability", 
    "customerNo", "maxVal","customerScore","accountLifetime","payments","age",
    "address","lastOrder"
]


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


def preprocess_input(df_test):
    ## convert '?' into -99
    ## reformat data types
    for col in df_test.columns:
        ## replace missing value with a numeric value, e.g., -99
        mask = (df_test[col] == "?")
        df_test.loc[mask, col] = -99

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
        df_test[col] = df_test[col].astype(float)

    condition_list = [
        (df_test["onlineStatus"] == "y"),
        (df_test["onlineStatus"] == "n"),
    ]
    choice_list = [1, 0]
    df_test["onlineStatus"] = np.select(
        condition_list, choice_list, default=df_test["onlineStatus"]
    )

    df_test["address"] = df_test["address"].astype(int)

    ## remove ID columns
    df_test.drop(labels=["sessionNo", "customerNo"], axis=1, inplace=True)

    ## determine time of day
    df_test = determine_time_of_day(df=df_test, column_name="startHour")

    ## one hot encoding
    selected_feature_list = ["availability", "address", "time_of_day", "onlineStatus"]
    for selected_feature in selected_feature_list:
        df_dummy_values = pd.get_dummies(df_test[selected_feature])
        df_dummy_values.columns = [
            selected_feature.replace(" ", "_") + "_" + str(col)
            for col in df_dummy_values.columns
        ]
        df_test = pd.concat([df_test, df_dummy_values], axis=1)
        ## remove redundant features - we've done one hot encoding
        df_test.drop(labels=selected_feature, axis=1, inplace=True)

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
        try:
            df_test[col] = df_test[col].astype(int)
        except:
            df_test[col] = 0

    return df_test


def model_fn(model_dir):
    log.info(f"In model_fn. Model directory is {model_dir}")

    model_file_path = os.path.join(model_dir, "model.pkl")

    log.info(f"Loading the model from {model_file_path}")
    with open(model_file_path, "rb") as f:
        model_clf = pickle.load(f)

    log.info(f"Model is successfully loaded from {model_file_path}")

    return model_clf


def input_fn(request_body, content_type):
    assert content_type in [
        JSON_CONTENT_TYPE,
        CSV_CONTENT_TYPE,
    ], f"Request has an unsupported ContentType in content_type: {content_type}"

    log.info(f"Request body CONTENT-TYPE is: {content_type}")
    log.info(f"Request body TYPE is: {type(request_body)}")

    log.info("Deserializing the input data.")
    log.info(f"Request body is: {request_body}")

    if content_type == JSON_CONTENT_TYPE:
        ## convert input json object as a dataframe of one row
        request = json.loads(request_body)
        log.info(f"Loaded JSON object: {request}")
        df_test = pd.DataFrame(request["data"], index=[0])
    elif content_type == CSV_CONTENT_TYPE:
        # data = request_body.decode('utf-8')
        # s = StringIO.StringIO(data)
        s = StringIO(request_body)
        df_test = pd.read_csv(s, header=None)
        df_test.columns = DEFAULT_COLUMN_NAME_LIST

    df_test = preprocess_input(df_test=df_test)
    return df_test


# inference
def predict_fn(input_object, model):
    log.info("In predict_fn")

    log.info("Calling model")
    feature_name_list = model.feature_names
    prediction = model.predict(input_object[feature_name_list])
    
    log.info(f"Prediction results:\n{prediction}")

    return prediction

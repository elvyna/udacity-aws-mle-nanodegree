import json
import logging
import sys
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

logging.basicConfig(
    format="%(filename)s %(asctime)s %(levelname)s Line no: %(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

log.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'

def model_fn(model_dir):
    log.info(f"In model_fn. Model directory is {model_dir}")
    
    model_file_path = os.path.join(model_dir, "model.pkl")
    
    log.info(f"Loading the model from {model_file_path}")
    with open(model_file_path, "rb") as f:
        model_clf = pickle.load(f)
        
    log.info(f'Model is successfully loaded from {model_file_path}')

    return model_clf

def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    assert content_type == JSON_CONTENT_TYPE, f"Request has an unsupported ContentType in content_type: {content_type}"
    log.info('Deserializing the input data.')

    ## process image that are passed to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    log.debug(f"Request body CONTENT-TYPE is: {content_type}")
    log.debug(f"Request body TYPE is: {type(request_body)}")
    
    log.debug(f"Request body is: {request_body}")
    request = json.loads(request_body)
    log.debug(f"Loaded JSON object: {request}")

    ## TO DO

    # url = request['url']
    # img_content = requests.get(url).content
    # return Image.open(io.BytesIO(img_content))

# inference
def predict_fn(input_object, model):
    log.info("In predict_fn")
    ## TO DO: transform input
    log.info("Transforming input")
    # input_object = test_transform(input_object)
    
    log.info("Calling model")
    prediction = model(input_object.unsqueeze(0))

    return prediction